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

# 4x6 derivative of exp([ẟ])*x wrt ẟ
def dxyzw_dse3(xyzw):
    x = xyzw[0,0]
    y = xyzw[1,0]
    z = xyzw[2,0]
    w = xyzw[3,0]
    result = np.array([
        [0, z, -y, w, 0, 0],
        [-z, 0, x, 0, w, 0],
        [y, -x, 0, 0, 0, w],
        [0, 0, 0, 0, 0, 0]
        ])
    return result;

def apply_camera_matrix(fxfycxcy, xyzw):
    """Applies the pinhole camera model
    fx, fy, cx, cy to the camera-frame point
    xyzw, and returns the resulting image pixel
    coordinates and derivatives
    """
    xyzw = xyzw.flatten()
    fxfycxcy = fxfycxcy.flatten()

    x = xyzw[0];
    y = xyzw[1];
    z = xyzw[2];

    fx = fxfycxcy[0];
    fy = fxfycxcy[1];
    cx = fxfycxcy[2];
    cy = fxfycxcy[3];

    cam_x = fx * x/z + cx;
    cam_y = fy * y/z + cy;

    xy = np.array([[cam_x, cam_y]]).T

    dxy_dcamparams = np.array([
        [x/z,   0, 1, 0],
        [  0, y/z, 0, 1]
    ])

    dxy_dxyzw = np.array([
        [fx/z,    0, -fx * x/(z*z), 0],
        [   0, fy/z, -fy * y/(z*z), 0]
    ])

    return xy, dxy_dcamparams, dxy_dxyzw


def project_points(camparams,
                   se3_world_camera,
                   se3_world_object, 
                   object_xyzws):
    
    tx_camera_world = se3_exp(-se3_world_camera)
    tx_world_object = se3_exp(se3_world_object)
    
    jl_object = se3_left_jacobian(se3_world_object)
    jl_neg_camera = se3_left_jacobian(-se3_world_camera)

    num_object_points = object_xyzws.shape[1]

    dxys_dcamera = np.empty((num_object_points*2, 6), dtype=float)
    dxys_dobject = np.empty((num_object_points*2, 6), dtype=float)
    xys = np.empty((num_object_points*2, 1), dtype=float)
    
    for c in range(num_object_points):
        object_xyzw = object_xyzws[:, c:c+1]
        world_xyzw = tx_world_object @ object_xyzw
        camera_xyzw = tx_camera_world @ world_xyzw

        # consider perturbation of tx_world_object and its effect on the campoint
        # tx_cam_world * exp(se3_world_object + delta) * object_xyzw
        # tx_cam_world * exp(Jl * delta) * exp(se3_world_object) * object_xyzw
        # tx_cam_world * dxyzw_dse3[exp(se3_world_object) * object_xyzw] * Jl * delta
        # \________________________________ ________________________________/
        #                                  v
        #                                dcampoint_dobject
        dcampoint_dobject = tx_camera_world @ dxyzw_dse3(world_xyzw) @ jl_object

        # consider perturbation of tx_world_camera and its effect on the campoint
        # exp(-(se3_world_cam + delta)) * tx_world_object * object_xyzw
        # exp(-se3_world_cam + -delta)) * tx_world_object * object_xyzw
        # exp(Jl * -delta) tx_cam_world * tx_world_object * object_xyzw
        # dxyzw_dse3[tx_cam_world * tx_world_object * object_xyzw] * Jl * -delta
        # \________________________________ ________________________________/
        #                                  v
        #                                dcampoint_dcamera
        dcampoint_dcamera = -dxyzw_dse3(camera_xyzw) @ jl_neg_camera

        xy, dxy_dcamparams, dxy_dcampoint = apply_camera_matrix(camparams, camera_xyzw);
        dxy_dobject = dxy_dcampoint @ dcampoint_dobject
        dxy_dcamera = dxy_dcampoint @ dcampoint_dcamera

        dxys_dcamera[2*c:2*c+2,:] = dxy_dcamera
        dxys_dobject[2*c:2*c+2,:] = dxy_dobject
        xys[2*c:2*c+2,:] = xy

    return xys, dxys_dcamera, dxys_dobject

def project(camparams, 
            se3_world_camera,
            se3_world_object, 
            object_xyzw):

    tx_camera_world = se3_exp(-se3_world_camera)
    tx_world_object = se3_exp(se3_world_object)
    
    campoint = tx_camera_world @ tx_world_object @ object_xyzw

    # consider perturbation of tx_world_object and its effect on the campoint
    # tx_cam_world * exp(se3_world_object + delta) * object_xyzw
    # tx_cam_world * exp(Jl * delta) * exp(se3_world_object) * object_xyzw
    # tx_cam_world * dxyzw_dse3[exp(se3_world_object) * object_xyzw] * Jl * delta
    # \________________________________ ________________________________/
    #                                  v
    #                                dcampoint_dobject
    dcampoint_dobject = tx_camera_world @ dxyzw_dse3(tx_world_object @ object_xyzw) @ se3_left_jacobian(se3_world_object)

    # consider perturbation of tx_world_camera and its effect on the campoint
    # exp(-(se3_world_cam + delta)) * tx_world_object * object_xyzw
    # exp(-se3_world_cam + -delta)) * tx_world_object * object_xyzw
    # exp(Jl * -delta) tx_cam_world * tx_world_object * object_xyzw
    # dxyzw_dse3[tx_cam_world * tx_world_object * object_xyzw] * Jl * -delta
    # \________________________________ ________________________________/
    #                                  v
    #                                dcampoint_dcamera
    dcampoint_dcamera = -dxyzw_dse3(tx_camera_world @ tx_world_object @ object_xyzw) @ se3_left_jacobian(-se3_world_camera)

    # perturbing the camera (in world frame) has the opposite effect as perturbing an object
    # dcampoint_dcamera = -dcampoint_dobject
    
    xy, dxy_dcamparams, dxy_dcampoint = apply_camera_matrix(camparams, campoint);
    dxy_dobject = dxy_dcampoint @ dcampoint_dobject
    dxy_dcamera = dxy_dcampoint @ dcampoint_dcamera

    return xy, dxy_dcamera, dxy_dobject, dcampoint_dobject

