import numpy as np

FLIP_CAM = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

FLIP_WORLD = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def flip_tx_world_cam(tx_world_cam):
    return FLIP_WORLD @ tx_world_cam @ FLIP_CAM

def flip_tx_cam_world(tx_cam_world):
    return FLIP_CAM @ tx_cam_world @ FLIP_WORLD

def heuristic_flip_tx_world_cam(tx_world_cam):
    if tx_world_cam[2,3] < 0:
        return flip_tx_world_cam(tx_world_cam)
    else:
        return tx_world_cam

def heuristic_flip_tx_cam_world(tx_cam_world):
    if tx_cam_world[2,3] < 0:
        return FLIP_CAM @ tx_cam_world @ FLIP_WORLD
    else:
        return tx_cam_world
        
