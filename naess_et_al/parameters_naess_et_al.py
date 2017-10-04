import numpy as np

# All numbers in cm
dipole_loc = 7.8
brain_rad = 7.9
csftop_rad = 8.
skull_rad = 8.5
scalp_rad = 9.

sigma_brain = 1. / 300.  # S / cm
sigma_scalp = sigma_brain
sigma_csf = 5 * sigma_brain
sigma_skull20 = sigma_brain / 20.
sigma_skull40 = sigma_brain / 40.
sigma_skull80 = sigma_brain / 80.

# from gmsh sphere_4.geo
whitemattervol = 32
graymattervol = 64
csfvol = 96
skullvol = 128
scalpvol = 160

# measument points
# theta = np.arange(0, 180)
# phi_angle = 0 # -90 to 90

theta, phi_angle = np.mgrid[0:180:10, -90:90:10]
theta = theta.flatten()
phi_angle = phi_angle.flatten()

theta_r = np.deg2rad(theta)
phi_angle_r = np.deg2rad(phi_angle)

rad_tol = 0. #1e-2
x_points = (scalp_rad - rad_tol) * np.sin(theta_r) * np.cos(phi_angle_r)
y_points = (scalp_rad - rad_tol) * np.sin(theta_r) * np.sin(phi_angle_r)
z_points = (scalp_rad - rad_tol) * np.cos(theta_r)

ele_coords = np.vstack((x_points, y_points, z_points)).T

# dipole location - Radial
rad_dipole = {'src_pos': [0., 0., 7.85],
              'snk_pos': [0., 0., 7.75],
              'name': 'rad'}

# # dipole location - Tangential
tan_dipole = {'src_pos': [0., -0.05, 7.8],
              'snk_pos': [0., 0.05, 7.8],
              'name': 'tan'}

# # # dipole location - Mix
mix_dipole = {'src_pos': [0., -0.0353, 7.835],
              'snk_pos': [0., 0.0353, 7.764],
              'name': 'mix'}

dipole_list = [rad_dipole, tan_dipole, mix_dipole]
