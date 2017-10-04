# Define a tensorflow graph that generates data on the fly. 
# Each data example belongs to one of two classes:
#    * background EEG
#    * 'Zone' EEG: For each sample, a center source has been
#      chosen and all sources in a small (parameter) radius
#      around this source have been given the same activity
#      (temporal and dipole direction)

# To keep it compact this code does not have many comments,
# For more description see devel/DEVEL_synth_eeg_tf.py


# TODO: Merge RESHAPES, TRANSPOSES...
# TODO: merge source generation in main part and different_zone_distrib
# (use a same common func)


import numpy as np
import tensorflow as tf

from . import mix_mat

r_geom = mix_mat.r_geom
sigmas = mix_mat.sigmas
elec_10_20_coords = mix_mat.elec_10_20_coords


def create_synth_data_graph(possible_center_idxs,
                            n_totalsphere_sources,
                            r_sources, 
                            batch_size, 
                            in_size, 
                            r_zone, 
                            radial_only,
                            fixed_ampl,
                            different_zone_distrib,
                            r_geom,
                            sigmas):
    # possible_center_idxs: a tf.placeholder (see 2))
    
    # NOTE: a good default is to use 500 totalsphere sources ?
    # 5000 totalsphere sources will saturate 6 GB of memory. 

    # =========== FIRST, all numpy code =====================

    # ----- Coordinates of electrodes
    coords_elec = np.zeros((len(elec_10_20_coords), 3))
    coords_elec[:, 0] = \
        [r_geom['scalp']] * len(elec_10_20_coords) # r
    coords_elec[:, 1] = np.deg2rad(
        [v[0] for v in elec_10_20_coords.values()]) # theta
    coords_elec[:, 2] = np.deg2rad(
        [v[1] for v in elec_10_20_coords.values()]) # phi
    # Here we keep spherical coords in rads
    n_elec = coords_elec.shape[0]

    # ----- Coordinates of sources
    coords_sources = mix_mat.get_spread_points_on_sphere(
        n=n_totalsphere_sources, r=r_sources, return_cartesian=False)
    n_sources = len(coords_sources)
    print(" ---- Number of sources after keep \'cortex\' only: ", \
          n_sources)

    # ----- Get mixing matrices source_activity -> electrode potential
    m_radial, m_tangential_x, m_tangential_y = \
    mix_mat.get_mixmats(
        coords_elec, coords_sources, r_geom, sigmas, n_legendre=100)
    m_all = np.stack(
        [m_radial, m_tangential_x, m_tangential_y]).astype(np.float32)
    # shape (3, n_sources, n_elec)

    # ----- Pairwise distances between sources
    coords_sources = mix_mat.spherical_to_cart(coords_sources)
    coords_sources_1 = np.reshape(
        coords_sources, [n_sources, 3, 1])
    coords_sources_2 = np.reshape(
        np.transpose(coords_sources), [1, 3, n_sources])
    coords_diff = coords_sources_2 - coords_sources_1
    dist22_sources = np.sqrt(
        1e-8 + np.sum(np.square(coords_diff), axis=1))
    # (n_sources, n_sources)

    # ----- For each source, get sources within r_zone of it
    condition = dist22_sources <= r_zone
    avg_nb_neighbors = np.mean(np.sum(condition, axis=1))
    print('--------------------------------------------------------')
    print(('Average number (over sources) of sources within a \n'
           'distance %.2f cm of a source (including itself) : %.2f' 
           %(r_zone, avg_nb_neighbors)))
    print('--------------------------------------------------------')

    # ----- Idxs of neighbor sources, for each source
    # note: not all sources have the same number of neighbors
    neighbors_list = [np.where(condition[i]==True)[0] \
                      for i in range(n_sources)]
    # ----- Number of neighbor sources, for each source
    nb_neighbors_list = [len(neighbors_list[i]) \
                         for i in range(n_sources)]


    def get_idxs(centralsource_idxs, n_sources):
        # get list of indices (and associated source numbers)
        # where source should be modified in act_sources_flat. 

        # We will use this NUMPY function by calling it with a tf.py_func
        # It will run on CPU which is not great. But it should be fast 
        # enough because there is only indexing (no calculus) and data 
        # transfer gpu-> cpu is light (centralsource_idxs.shape=(batch_size,)

        # Note: because the entries of neighbors_list and nb_neighbors_list
        # have different lengths, we cannot use normal vectorized indexing
        # and have to use list comprensions...

        # list of idxs to modify in act_sources_flat
        idxs_to_modify = np.concatenate(
            [np.array(neighbors_list[centralsource_idxs[k]]) \
             + k * n_sources \
            for k in range(batch_size)], 
            axis=0).astype(np.int32)

        # and corresponding sources to insert
        idxs_sources = np.concatenate(
            [np.array([k] * nb_neighbors_list[centralsource_idxs[k]]) \
            for k in range(batch_size)],
            axis=0).astype(np.int32)

        return idxs_to_modify, idxs_sources


    # ============ AND THEN, the TF GRAPH ==================
    # --- 0) Common part --- :
    # activity = ampl. * sin(w.x + p)
    # w = 2 * pi * f / 128 and f=128Hz
    # f drawn uniformly between 1 and 32Hz
    # amplitude in uniform([[0.2, 1.]])
    # phase in uniform([[0, 2pi]])
    if fixed_ampl:
        ampl = tf.ones([batch_size, n_sources, 1])
    else:
        ampl = tf.random_uniform(
            [batch_size, n_sources, 1], 0.2, 1.)
    f = tf.random_uniform(
        [batch_size, n_sources, 1], 1., 32.)
    w = tf.multiply(f, 2*np.pi/128.)
    p = tf.random_uniform(
        [batch_size, n_sources, 1], 0., 2*np.pi)
    x = tf.constant(np.arange(in_size).astype(np.float32))
    x = tf.reshape(x, [1, 1, in_size])
    act_sources = tf.multiply(
        ampl, tf.sin(tf.multiply(w, x) + p))
    # (batch_size, n_sources, in_size)
    # draw random 3d directions of the dipoles
    if radial_only:
        a_r = tf.ones([batch_size, n_sources, 1, 1])
        a_thetaphi = tf.zeros([batch_size, n_sources, 1, 1])
        a_rtp = tf.concat([a_r, a_thetaphi, a_thetaphi], axis=-1)
    else:
        zxy = tf.random_normal([batch_size, n_sources, 1, 3])
        r = tf.sqrt(1e-8 + tf.reduce_sum(
            tf.square(zxy), axis=-1, keep_dims=True))
        a_rtp = tf.divide(zxy, r)
    act_sources = tf.reshape(act_sources, 
        [batch_size, n_sources, in_size, 1])
    act_sources_rtp = tf.multiply(act_sources, a_rtp)
    # (batch_size, n_sources, in_size, 3)
    # REM: this tensor can be BIG in memory !

    # --- 1) Class 'background' --- :
    act_sources_rtp_clb = act_sources_rtp
    # prepare for matmul
    act_sources_rtp_clb = tf.transpose(
        act_sources_rtp_clb, [3, 1, 0, 2])
    act_sources_rtp_clb = tf.reshape(
        act_sources_rtp_clb, [3, n_sources, batch_size*in_size])
    act_sources_rtp_clb = tf.transpose(
        act_sources_rtp_clb, [0, 2, 1])
    # And multiply by m_all to get electrode potentials
    M_all = tf.constant(m_all)
    act_elec_zxy_clb = tf.matmul(
        act_sources_rtp_clb, M_all)
    act_elec_zxy_clb = tf.transpose(
        act_elec_zxy_clb, [0, 2, 1])
    act_elec_zxy_clb = tf.reshape(
        act_elec_zxy_clb, [3, n_elec, batch_size, in_size])
    act_elec_zxy_clb = tf.transpose(
        act_elec_zxy_clb, [2, 3, 0, 1]) 
    act_elec_clb = tf.reduce_sum(act_elec_zxy_clb, axis=2)
    #(batch_size, in_size, n_elec)
    # normalize
    m, v = tf.nn.moments(act_elec_clb, axes=[1], keep_dims=True)
    v_all = tf.reduce_mean(v, axis=2, keep_dims=True)
    act_elec_clb = tf.divide(
        (act_elec_clb - m), 2*tf.sqrt(1e-8 + v_all))

    # --- 2) Class 'common zone' --- :
    # placeholder to optionally specify a subset of all sources
    # amongst which to draw the common zone centers. 
    # (Originally in graph, here we get it from function args)
    #possible_center_idxs = tf.placeholder_with_default(
    #    np.arange(n_sources).astype(np.int32), shape=[])
    # indices (amongst possible_centers) of the sources
    # which will be centers:
    n_possible_centers = tf.shape(possible_center_idxs)[0] # dynamic shape
    csource_idxs_amgst_possible = tf.random_uniform(
        [batch_size], 0, n_possible_centers, dtype=tf.int32)
    centralsource_idxs = tf.gather(
    possible_center_idxs, csource_idxs_amgst_possible)
    # initially, start with background EEG
    act_sources_rtp_clz = act_sources_rtp
    # flatten so that all modifs can be done on axis 0
    act_sources_flat = tf.transpose(act_sources_rtp_clz, [2, 3, 0, 1])
    act_sources_flat = tf.reshape(
        act_sources_flat, [in_size, 3, batch_size*n_sources])
    act_sources_flat = tf.transpose(act_sources_flat, [2, 0, 1])
    # now (batch_size*n_sources, in_size, 3)
    # make it a tf.Variable (because tf.scatter_update needs this)
    act_sources_flat_var = tf.get_variable(
        'act_sources_flat', 
        initializer= \
          np.zeros((batch_size*n_sources, in_size, 3)).astype(np.float32),
        trainable=False)
    # this variable has an initial value, but we also need to assign
    # a new value to it each time we want to use it anew. 
    assign_act_sources_flat_var = tf.assign(
        act_sources_flat_var, act_sources_flat)
    # zip indices:
    b_range = tf.constant(np.arange(batch_size), dtype=tf.int32)
    indices_bs = tf.stack([b_range, centralsource_idxs], axis=1)
    # if different_zone_distrib, draw NEW sources with a waveform that is
    # different from the original sine. To assign it to new 'zone'
    if different_zone_distrib:
        # let's use a sawtooth signal here
        if fixed_ampl:
            ampl = tf.ones([batch_size, 1])
        else: 
            ampl = tf.random_uniform(
                [batch_size, 1], 0.2, 1.)
        f = tf.random_uniform(
            [batch_size, 1], 1., 32.)
        p = tf.random_uniform(
            [batch_size, 1], 0., 1.)
        x = tf.constant(np.arange(in_size).astype(np.float32))
        x = tf.reshape(x, [1, in_size])
        selected_sources = tf.multiply(
            ampl, tf.multiply(f/128., x) + p \
                  - tf.floor(tf.multiply(f/128., x) + p) - 0.5)
        # draw random 3d directions
        if radial_only:
            a_r = tf.ones([batch_size, 1, 1])
            a_thetaphi = tf.zeros([batch_size, 1, 1])
            a_rtp = tf.concat([a_r, a_thetaphi, a_thetaphi], axis=-1)
        else:
            zxy = tf.random_normal([batch_size, 1, 3])
            r = tf.sqrt(1e-8 + tf.reduce_sum(
                tf.square(zxy), axis=-1, keep_dims=True))
            a_rtp = tf.divide(zxy, r)
        selected_sources = tf.reshape(selected_sources,
            [batch_size, in_size, 1])
        selected_sources = tf.multiply(selected_sources, a_rtp)
        # (batch_size, in_size, 3)
    # otherwise, select the signals from the selected centers
    else:
        selected_sources = tf.gather_nd(
            act_sources_rtp_clz, indices_bs)
    # remaining shape (batch_size, in_size, 3)
    # get graph nodes for idxs_to_modify, idxs_sources using a py_func: 
    idxs_to_modify, idxs_sources = tf.py_func(
        get_idxs, 
        inp=[centralsource_idxs, n_sources], 
        Tout=[tf.int32, tf.int32])
    # 'Copy' each of the selected_sources as many times as we need
    # to insert it:
    sources_to_insert = tf.gather(
        selected_sources, idxs_sources)

    # Make sure act_sources_flat_var gets a new value
    # before we partially update it and use it for end data:
    with tf.control_dependencies([assign_act_sources_flat_var]):
        # update the sources which are in the 'common zone':
        act_sources_flat_var = tf.scatter_update(
            act_sources_flat_var,
            indices=idxs_to_modify,
            updates=sources_to_insert)
        # (batch_size*n_sources, in_size, 3)
        # reshape for multiplication by m_all
        act_sources_rtp_clz = tf.transpose(
            act_sources_flat_var, [1, 2, 0])
        act_sources_rtp_clz = tf.reshape(
            act_sources_rtp_clz, [in_size, 3, batch_size, n_sources])
        act_sources_rtp_clz = tf.transpose(
            act_sources_rtp_clz, [1, 3, 2, 0])
        act_sources_rtp_clz = tf.reshape(
            act_sources_rtp_clz, [3, n_sources, batch_size*in_size])
        act_sources_rtp_clz = tf.transpose(
            act_sources_rtp_clz, [0, 2, 1])
        # (3, batch_size*in_size, n_sources)
        act_elec_zxy_clz = tf.matmul(
            act_sources_rtp_clz, M_all)
        # (3, batch_size*in_size, n_elec)
        act_elec_zxy_clz = tf.transpose(
            act_elec_zxy_clz, [0, 2, 1])
        act_elec_zxy_clz = tf.reshape(
            act_elec_zxy_clz, [3, n_elec, batch_size, in_size])
        act_elec_zxy_clz = tf.transpose(
            act_elec_zxy_clz, [2, 3, 0, 1])
        act_elec_clz = tf.reduce_sum(act_elec_zxy_clz, axis=2) 
        # (batch_size, in_size, n_elec)
        # normalize
        m, v = tf.nn.moments(act_elec_clz, axes=[1], keep_dims=True)
        v_all = tf.reduce_mean(v, axis=2, keep_dims=True)
        act_elec_clz = tf.divide(
            (act_elec_clz - m), 2*tf.sqrt(1e-8 + v_all))

    # REM: csource_idxs_amgst_possible is the class number
    # amongst 2)='zone' classes
    return act_elec_clb, act_elec_clz, \
        csource_idxs_amgst_possible











