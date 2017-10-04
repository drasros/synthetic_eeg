# Tests of various functions in mix_mat.py
# for sanity check before using the simulator

import numpy as np
import time
import os
import mix_mat

#################################################################
# CHECK change of coordinates
# draw a few random points with theta in [0, 180](deg) and phi in [-180, 180](deg)
# transform spherical to cart and reverse, check that combination is the identity
rs = np.ones(shape=[50])
thetas = np.random.uniform(0., 180., size=[50])
thetas = np.deg2rad(thetas)
phis = np.random.uniform(-180., 180., size=[50])
phis = np.deg2rad(phis)
points = np.stack([rs, thetas, phis], axis=1)

points_new = mix_mat.spherical_to_cart(mix_mat.cart_to_spherical(points))
assert (np.absolute(points_new - points) < 1e-10).all()

#################################################################
# CHECK Rotation matrices 
# draw vectors a, b and check that R(a->b) * R(b->a) = I
a = np.random.uniform(-1., 1., size=[100, 3])
b = np.random.uniform(-1., 1., size=[50, 3])

R = mix_mat.get_rotation_matrix(a, b) # shape (100, 50, 3, 3)
R = np.squeeze(R)
R_ = mix_mat.get_rotation_matrix(b, a) # shape (50, 100, 3, 3)
R_ = np.squeeze(R_)
R_ = np.swapaxes(R_, 0, 1)

I = np.reshape(np.eye(3), (1, 1, 3, 3))
assert (np.absolute(np.matmul(R, R_) - I) < 1e-10).all()

#################################################################
# CHECK get_elec_coords_in_sources_ref
coords_sources = np.array([[1., 0., 0.]])
coords_elec = np.array([[1., 0., 0.], [np.pi/4., np.pi/4., np.pi/4.]])

# let's use a point on the z axis as source
# then coords_elec in source ref should equal coords_elec
# if so, the function get_elec_coords_in_sources_ref will also work 
# for coords_sources, because the only difference will be the input 
# provided to get_rotation_matrix, and we have already checked that 
# get_rotation_matrix works properly. 

coords_elec_ref_source = mix_mat.get_elec_coords_in_sources_ref(
    coords_elec, coords_sources)
assert(np.absolute(coords_elec - coords_elec_ref_source) < 1e-10).all()

#################################################################
# CHECK r_list

# geometry
r_geom = mix_mat.r_geom

# Sources ---------
# one unique source located on z axis at r=7.8
# (just below the top of the cortex)
dipole_loc = 7.8
r_source = np.array([dipole_loc])
theta_source = np.array([0])
theta_source = np.deg2rad(theta_source)
phi_source = np.array([0])
phi_source = np.deg2rad(phi_source)
coords_source = np.stack([r_source, theta_source, phi_source], axis=1)

r_list = mix_mat.get_r_ij(coords_source[:, 0], r_geom)
r_list_o = np.load(os.path.join('naess_et_al', 'r_list.npy'))

r_list_ = [(elmt[0] if type(elmt)==np.ndarray else elmt)
           for elmt in r_list]
assert(np.absolute(r_list_ - r_list_o) < 1e-10).all()

#################################################################
# CHECK get_V_Y_Z
# execute 'python analytical_naess_et_al.py' first to calculate
# V(n), Y(n), Z(n) from original article. 

# sigmas
sigmas = mix_mat.sigmas
# Electrodes --------
# same as in paper
# we consider only electrodes on the surface of the scalp
theta_elec, phi_elec = np.mgrid[0:180:10, -90:90:10]
theta_elec = theta_elec.flatten()
phi_elec = phi_elec.flatten()
theta_elec = np.deg2rad(theta_elec)
phi_elec = np.deg2rad(phi_elec)
n_elec = len(theta_elec)
r_elec = [r_geom['scalp'] for _ in range(n_elec)] # r_geom['scalp']-1e-2
coords_elec = np.stack([r_elec, theta_elec, phi_elec], axis=1)


n_legendre = 100
n = np.arange(1, n_legendre)

s1, s2, s3 = mix_mat.get_sigmas_ij(sigmas)

V, Y, Z = mix_mat.get_V_Y_Z(n, r_list, sigmas)
V_o = np.load(os.path.join('naess_et_al', 'V.npy'))
Y_o = np.load(os.path.join('naess_et_al', 'Y.npy'))
Z_o = np.load(os.path.join('naess_et_al', 'Z.npy'))

assert(V.squeeze() - V_o == 0).all()
assert(Y.squeeze() - Y_o == 0).all()
assert(Z.squeeze() - Z_o == 0).all()


#################################################################
# CHECK A4, B4

A4, B4 = mix_mat.get_A4_B4(n, r_list, sigmas, V, Y, Z)
A4_o = np.load(os.path.join('naess_et_al', 'A4.npy'))
B4_o = np.load(os.path.join('naess_et_al', 'B4.npy'))

assert(A4.squeeze() - A4_o == 0.).all()
assert(B4.squeeze() - B4_o == 0.).all()

#################################################################
# CHECK mixmats
# To check that the correct mixing matrices are obtained
# (and benchmark for speed), let's use our code and the original
# code on one unique dipole placed on the z axis as in the paper,
# (and same electrode positions)
# and check that we obtain the same results as with the code in 
# the orinal code

# number of legendre polynomials
n_legendre = 100

t = time.time()
m_radial, _, m_tangential = mix_mat.get_mixmats(
    coords_elec, coords_source, r_geom, sigmas, n_legendre)
# we use only m_tangential_y because that's the dipole orientation
# they use in the paper. tangential_x has exactly the same form
# except pi/2 offset for phi!

# for potential, use same p values as in article
pot_radial = 0.1 * m_radial
pot_tangential = - 0.1 * m_tangential
t = time.time() - t
print('Radial and tangential mixing matrices computed in %s seconds.' %str(t))

# now compare with original article
pot_rad_original = np.load(os.path.join('naess_et_al', 'Analytical_rad.npy'))
pot_tan_original = np.load(os.path.join('naess_et_al', 'Analytical_tan.npy'))

assert(np.absolute(
    np.squeeze(pot_radial) - pot_rad_original) < 1e-10).all()
assert(np.absolute(
    np.squeeze(pot_tangential) - pot_tan_original) < 1e-10).all()

print('------------------')
print('ALL TESTS PASSED. ')
print('------------------')


