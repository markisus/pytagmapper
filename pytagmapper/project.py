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

def project(camera_matrix, tx_camera_object, keypoints_mat):
    # keypoints_mat is expected to be (2, n)
    # where n is the number of keypoints
    num_kps = keypoints_mat.shape[1]
    
    camera_kps = tx_camera_object @ keypoints_mat
    dimage_kps_dcamera = np.empty((num_kps*2,6))

    image_kps = camera_matrix @ camera_kps[:3,:]
    # print("before homog\n", image_kps)
    # print("after homog\n", image_kps)

    for i in range(6):
        pert = np.zeros((6,1))
        pert[i,0] = 1
        pertmat = se3_to_matrix(pert)
        # print("pertmat", i, "is\n", pertmat)
        dimage_kps = -camera_matrix @ (pertmat @ camera_kps)[:3,:] # 3 x num_kps
        for kp_idx in range(num_kps):
            x = image_kps[0, kp_idx]
            y = image_kps[1, kp_idx]
            z = image_kps[2, kp_idx]

            dx = dimage_kps[0, kp_idx]
            dy = dimage_kps[1, kp_idx]
            dz = dimage_kps[2, kp_idx]

            # print("corner", kp_idx, "dx, dy, dz", dx, dy, dz)
            # dxh = dx
            # dyh = dy
            dxh = (1/z)*dx + (-x/z**2)*dz
            dyh = (1/z)*dy + (-y/z**2)*dz
            dimage_kps_dcamera[  2*kp_idx, i] = dxh
            dimage_kps_dcamera[2*kp_idx+1, i] = dyh
            
    dimage_kps_dobject = dimage_kps_dcamera @ -SE3_adj(tx_camera_object)

    for i in range(num_kps):
        image_kps[:,i] /= image_kps[2,i] # homogenize
    image_kps = np.reshape(image_kps[:2,:], (num_kps*2,1), 'F')

    return image_kps, dimage_kps_dcamera, dimage_kps_dobject

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

    

    
    

