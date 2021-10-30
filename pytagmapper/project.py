import numpy as np
from pytagmapper.geometry import *

def get_corners_mat(size):
    mat = np.array([
        [-0.5, 0.5, 0.5,-0.5],
        [ 0.5, 0.5,-0.5,-0.5],
        [   0,   0,   0,   0],
        [   1,   1,   1,   1],
    ], np.float64)
    mat[:3,:] *= size
    return mat

def get_corners_mat2d(size):
    mat = np.array([
        [-0.5, 0.5, 0.5,-0.5],
        [ 0.5, 0.5,-0.5,-0.5],
        [   1,   1,   1,   1],
    ], np.float64)
    mat[:2,:] *= size
    return mat

def project(camera_matrix, tx_camera_tag, corners_mat):
    camera_corners = tx_camera_tag @ corners_mat
    dimage_corners_dcamera = np.empty((8,6))

    image_corners = camera_matrix @ camera_corners[:3,:]
    # print("before homog\n", image_corners)
    # print("after homog\n", image_corners)

    for i in range(6):
        pert = np.zeros((6,1))
        pert[i,0] = 1
        pertmat = se3_to_matrix(pert)
        # print("pertmat", i, "is\n", pertmat)
        dimage_corners = -camera_matrix @ (pertmat @ camera_corners)[:3,:] # 3 x 4
        for corner_idx in range(4):
            x = image_corners[0, corner_idx]
            y = image_corners[1, corner_idx]
            z = image_corners[2, corner_idx]

            dx = dimage_corners[0, corner_idx]
            dy = dimage_corners[1, corner_idx]
            dz = dimage_corners[2, corner_idx]

            # print("corner", corner_idx, "dx, dy, dz", dx, dy, dz)
            # dxh = dx
            # dyh = dy
            dxh = (1/z)*dx + (-x/z**2)*dz
            dyh = (1/z)*dy + (-y/z**2)*dz
            dimage_corners_dcamera[  2*corner_idx, i] = dxh
            dimage_corners_dcamera[2*corner_idx+1, i] = dyh
            
    dimage_corners_dtag = dimage_corners_dcamera @ -SE3_adj(tx_camera_tag)

    for i in range(4):
        image_corners[:,i] /= image_corners[2,i] # homogenize
    image_corners = np.reshape(image_corners[:2,:], (8,1), 'F')

    return image_corners, dimage_corners_dcamera, dimage_corners_dtag

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    camera_matrix = np.array([
        [100,   0, 50],
        [  0, 100, 50],
        [  0,   0,  1]
    ])
    tx_camera_tag = np.array([
        [1, 0, 0,   0],
        [0, -1, 0,  0],
        [0, 0, -1,0.3],
        [0, 0,  0,  1],
    ])
    corners_mat = get_corners_mat(0.3)

    spicy = rng.random((6,1))
    tx_camera_tag = tx_camera_tag @ se3_exp(spicy * 0.1)
    
    epsilon = 1e-6
    # perturb = rng.random((6,1))
    perturb = np.array([[0,0,0,0,0,1]]).T

    image_corners, dimage_corners_dcamera, dimage_corners_dtag = project(camera_matrix, tx_camera_tag, corners_mat)

    tx_camera_tag_t = tx_camera_tag @ se3_exp(perturb * epsilon)
    tx_camera_tag_c = se3_exp(-perturb * epsilon) @ tx_camera_tag

    image_corners_t, _, _ = project(camera_matrix, tx_camera_tag_t, corners_mat)
    image_corners_c, _, _ = project(camera_matrix, tx_camera_tag_c, corners_mat)

    deriv_t_numerical = (image_corners_t - image_corners)/epsilon
    deriv_t = dimage_corners_dtag @ perturb
    print("deriv_t numerical\n",deriv_t_numerical)
    print("deriv_t\n",deriv_t)
    print("delta_t\n",np.linalg.norm(deriv_t - deriv_t_numerical))

    deriv_c_numerical = (image_corners_c - image_corners)/epsilon
    deriv_c = dimage_corners_dcamera @ perturb
    print("deriv_c numerical\n",deriv_c_numerical)
    print("deriv_c\n",deriv_c)
    print("delta_c\n",np.linalg.norm(deriv_c - deriv_c_numerical))

    

    
    

