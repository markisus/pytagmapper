import numpy as np
from geometry import *
from project import *
from data import *
from heuristics import *

class InsideOutTracker:
    def __init__(self, camera_matrix, map_data,
                 tx_world_viewpoint = None):
        self.camera_matrix = np.array(camera_matrix)
        self.tag_side_length = map_data['tag_side_length']
        self.corners_mat = get_corners_mat(self.tag_side_length)
        self.tag_locations = map_data['tag_locations']

        self.txs_world_tag = {}

        self.map_type = map_data['map_type']
        if self.map_type == '2.5d':
            for tag_id, xytz_world_tag in self.tag_locations.items():
                self.txs_world_tag[tag_id] = \
                    xytz_to_SE3(np.array([xytz_world_tag]).T)
        elif self.map_type == '2d':
            for tag_id, xyt_world_tag in self.tag_locations.items():
                self.txs_world_tag[tag_id] = \
                    xyt_to_SE3(np.array([xyt_world_tag]).T)
        else:
            raise RuntimeError("Unsupported map type", self.map_type)

        self.tx_world_viewpoint = tx_world_viewpoint
        if self.tx_world_viewpoint is None:
            init_dist = 10 * self.tag_side_length
            self.tx_world_viewpoint = \
                np.array([
                    [1,  0,  0, 0],
                    [0, -1,  0, 0],
                    [0,  0, -1, init_dist],
                    [0,  0,  0, 1]
                ])

        self.error = float('inf')
        self.regularizer = 1e5

    def get_projections(self):
        tag_ids = []
        tag_corners =[]

        for tag_id, tx_world_tag in self.txs_world_tag.items():
            tx_viewpoint_tag = SE3_inv(self.tx_world_viewpoint) @ tx_world_tag
            projected_corners, _, _ = project(self.camera_matrix, tx_viewpoint_tag, self.corners_mat)
            tag_ids.append(tag_id)
            tag_corners.append(projected_corners)

        return tag_ids, tag_corners


    def update(self, tag_ids, tag_corners, force_update = True):
        JtJ = np.zeros((6,6))
        rtJ = np.zeros((1,6))
        curr_error = 0

        for tag_id, corners in zip(tag_ids, tag_corners):
            tx_world_tag = self.txs_world_tag.get(tag_id, None)
            if tx_world_tag is None:
                continue

            tx_viewpoint_tag = SE3_inv(self.tx_world_viewpoint) @ tx_world_tag
            projected_corners, dcorners_dcamera, _ = project(self.camera_matrix, tx_viewpoint_tag, self.corners_mat)

            residual = projected_corners - corners
            JtJ += dcorners_dcamera.T @ dcorners_dcamera
            rtJ += residual.T @ dcorners_dcamera
            curr_error += (residual.T @ residual)[0,0]

        if curr_error > self.error:
            self.regularizer *= 25
        else:
            self.regularizer *= 0.5

        improved = curr_error < self.error

        self.regularizer = min(self.regularizer, 1e4)
        self.regularizer = max(self.regularizer, 1e-3)

        self.error = curr_error

        if improved or force_update:
            update = np.linalg.solve(JtJ + self.regularizer * np.eye(6), -rtJ.T)
            self.tx_world_viewpoint = heuristic_flip_tx_world_cam(self.tx_world_viewpoint @ se3_exp(update))

        return improved


if __name__ == "__main__":
    data_dir = "data1"
    camera_matrix = load_camera_matrix(data_dir)
    tag_map = load_map(data_dir)

    print("camera_matrix", camera_matrix)
    print("tag map", tag_map)

    tracker = InsideOutTracker(camera_matrix,
                               tag_map["tag_side_length"],
                               tag_map["tag_locations"])

    rng = np.random.default_rng(0)
    gt_tx_world_viewpoint = tracker.tx_world_viewpoint
    gt_tx_world_viewpoint = gt_tx_world_viewpoint @ se3_exp(rng.random((6,1)) * 0.1)

    # detect tag 7
    xyt_world_tag7 = tag_map["tag_locations"][7]
    xyt_world_tag7 = np.array([xyt_world_tag7]).T
    tx_world_tag7 = SE2_to_SE3(xyt_to_SE2(xyt_world_tag7))

    tx_viewpoint_tag7 = SE3_inv(gt_tx_world_viewpoint) @ tx_world_tag7
    print("tx_viewpoint_tag7", tx_viewpoint_tag7)
    print("corners_mat", tracker.corners_mat)
    print("camera_mat", tracker.camera_matrix)
    
    detected_corners, _, _ = project(tracker.camera_matrix,
                                     tx_viewpoint_tag7,
                                     tracker.corners_mat)

    while tracker.error > 1e-3:
        tracker.update([7], [detected_corners])
        print("err",tracker.error)

    print(tracker.tx_world_viewpoint - gt_tx_world_viewpoint)

    exit(0)
