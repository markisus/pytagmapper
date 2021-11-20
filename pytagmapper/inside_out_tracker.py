import numpy as np
from pytagmapper.geometry import *
from pytagmapper.project import *
from pytagmapper.data import *
from pytagmapper.heuristics import *

def look_at_origin(from_xyz, up_dir):
    from_xyz = np.array(from_xyz, dtype=np.float64)
    up_dir = np.array(up_dir, dtype=np.float64)
    look_dir = -from_xyz
    look_dir /= np.linalg.norm(look_dir)
    x = np.cross(look_dir, up_dir) # z cross (-y) = x
    y = np.cross(look_dir, x) # z cross x = y
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    result = np.empty((4,4))
    result[:3,0] = x
    result[:3,1] = y
    result[:3,2] = look_dir
    result[:3,3] = from_xyz
    result[3,:] = [0, 0, 0, 1]
    return result

# success of the tracker heavily depends on initialization
# initialization from one of these viewpoints generally will succeed
INIT_TXS_WORLD_VIEWPOINT = [
    # topdown views
    look_at_origin([0,0,1], [0,1,0]),
    look_at_origin([0,0,1], [0,-1,0]),
    look_at_origin([0,0,1], [1,0,0]),
    look_at_origin([0,0,1], [-1,0,0]),

    # view from left
    look_at_origin([1,0,0.5], [0,0,1]),
    look_at_origin([1,0,0.5], [0,0,-1]),
    look_at_origin([1,0,0.5], [0,1,0]),
    look_at_origin([1,0,0.5], [0,-1,0]),

    # view from top
    look_at_origin([0,1,0.5], [0,0,1]),
    look_at_origin([0,1,0.5], [0,0,-1]),
    look_at_origin([0,1,0.5], [1,0,0]),
    look_at_origin([0,1,0.5], [-1,0,0]),

    # view from right
    look_at_origin([-1,0,0.5], [0,0,1]),
    look_at_origin([-1,0,0.5], [0,0,-1]),
    look_at_origin([-1,0,0.5], [0,1,0]),
    look_at_origin([-1,0,0.5], [0,-1,0]),

    # view from bottom
    look_at_origin([0,-1,0.5], [0,0,1]),
    look_at_origin([0,-1,0.5], [0,0,-1]),
    look_at_origin([0,-1,0.5], [1,0,0]),
    look_at_origin([0,-1,0.5], [-1,0,0]),
]

class InsideOutTracker:
    def __init__(self, camera_matrix, map_data,
                 tx_world_viewpoint = None, max_regularizer = 1e9):
        self.tag_locations = map_data['tag_locations']        
        self.camera_matrix = np.array(camera_matrix)
        self.tag_side_lengths = map_data['tag_side_lengths']
        self.default_tag_side_length = self.tag_side_lengths['default']
        self.default_corners_mat = get_corners_mat(self.default_tag_side_length)
        self.corners_mats = {}
        for tag_id, tag_side_length in self.tag_side_lengths.items():
            self.corners_mats[tag_id] = get_corners_mat(tag_side_length)

        self.txs_world_tag = {}

        map_lift_3d(map_data)
        for tag_id, tx_world_tag in self.tag_locations.items():
            self.txs_world_tag[tag_id] = np.array(tx_world_tag)

        self.tx_world_viewpoint = tx_world_viewpoint
        if self.tx_world_viewpoint is None:
            init_dist = 10 * self.default_tag_side_length
            self.tx_world_viewpoint = \
                np.array([
                    [1,  0,  0, 0],
                    [0, -1,  0, 0],
                    [0,  0, -1, init_dist],
                    [0,  0,  0, 1]
                ])

        self.error = float('inf')

        self.max_regularizer = max_regularizer
        self.regularizer = self.max_regularizer        

        self.txs_world_viewpoint = [tx.copy() for tx in INIT_TXS_WORLD_VIEWPOINT]
        for tx in self.txs_world_viewpoint:
            tx[:3,3] *= self.default_tag_side_length * 10
        self.num_hypotheses = len(self.txs_world_viewpoint)
        self.errors = [float('inf') for _ in range(self.num_hypotheses)]
        self.regularizers = [self.max_regularizer for _ in range(self.num_hypotheses)]
        self.converged_guess = None
        self.best_guess = 0

    def get_corners_mat(self, tag_id):
        return self.corners_mats.get(tag_id, self.default_corners_mat)

    def get_projections(self, guess_idx=-1):
        tag_ids = []
        tag_corners =[]

        if guess_idx >= 0:
            tx_world_viewpoint = self.txs_world_viewpoint[guess_idx]
        else:
            tx_world_viewpoint = self.tx_world_viewpoint

        for tag_id, tx_world_tag in self.txs_world_tag.items():
            tx_viewpoint_tag = SE3_inv(tx_world_viewpoint) @ tx_world_tag
            projected_corners, _, _ = project(self.camera_matrix, tx_viewpoint_tag, self.get_corners_mat(tag_id))
            tag_ids.append(tag_id)
            tag_corners.append(projected_corners)

        return tag_ids, tag_corners

    def update_guess(self, guess_idx, tags, force_update = False):
        tx_world_viewpoint = self.txs_world_viewpoint[guess_idx]
        error = self.errors[guess_idx]
        regularizer = self.regularizers[guess_idx]
        
        JtJ = np.zeros((6,6))
        rtJ = np.zeros((1,6))
        curr_error = 0

        for tag_id, corners in tags:
            tx_world_tag = self.txs_world_tag.get(tag_id, None)
            if tx_world_tag is None:
                continue

            corners = np.array(corners).reshape((8,1))
            tx_viewpoint_tag = SE3_inv(tx_world_viewpoint) @ tx_world_tag
            projected_corners, dcorners_dcamera, _ = project(self.camera_matrix, tx_viewpoint_tag, self.get_corners_mat(tag_id))

            residual = projected_corners - corners
            JtJ += dcorners_dcamera.T @ dcorners_dcamera
            rtJ += residual.T @ dcorners_dcamera
            curr_error += (residual.T @ residual)[0,0]

        if curr_error > error:
            regularizer *= 25
        else:
            regularizer *= 0.5

        improved = curr_error < error

        regularizer = min(regularizer, self.max_regularizer)
        regularizer = max(regularizer, 1e-3)

        if improved or force_update:
            update = np.linalg.solve(JtJ + regularizer * np.eye(6), -rtJ.T)
            tx_world_viewpoint = tx_world_viewpoint @ se3_exp(update)
            # tx_world_viewpoint = heuristic_flip_tx_world_cam(tx_world_viewpoint @ se3_exp(update))

        error = curr_error

        self.txs_world_viewpoint[guess_idx] = tx_world_viewpoint
        self.regularizers[guess_idx] = regularizer
        self.errors[guess_idx] = error

    def update1(self, tags, force_update = False):
        if self.converged_guess is not None:
            self.update_guess(self.converged_guess, tags, force_update)
            best_guess = self.converged_guess
        else:
            for i in range(self.num_hypotheses):
                self.update_guess(i, tags, force_update)

            # report the tx with the best error
            best_guess = 0
            best_error = float('inf')
            for i, error in enumerate(self.errors):
                if error < best_error:
                    best_guess = i
                    best_error = error

            # heuristic to check convergence
            num_tags = len([t for t, c in tags if t in self.txs_world_tag])
            if num_tags >= 2:
                pt_error = best_error / (num_tags * 4)
                if pt_error <= 30: # px
                    self.converged_guess = best_guess

        self.error = self.errors[best_guess]
        self.tx_world_viewpoint = self.txs_world_viewpoint[best_guess]
        self.regularizer = self.regularizers[best_guess]
        self.best_guess = best_guess

    def update(self, tag_ids, tag_corners, force_update = False):
        return self.update1(list(zip(tag_ids, tag_corners)), force_update)

