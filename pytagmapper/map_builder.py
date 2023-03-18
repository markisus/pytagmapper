import numpy as np
import math
from pytagmapper.geometry import *
from pytagmapper.info_state import *
from pytagmapper.project import project_points, get_corners_mat
from pytagmapper.heuristics import *
from pytagmapper.min_heap import MinHeap
import cv2

def solvePnPWrapper(obj_points, img_points, camera_matrix):
    succ, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, None)
    if not succ:
        raise RuntimeError("solvePnP failed")
    rot, _ = cv2.Rodrigues(rvec)
    tx_camera_obj = np.eye(4, dtype=np.float64)
    tx_camera_obj[:3,:3] = rot
    tx_camera_obj[:3,3:4] = tvec
    return tx_camera_obj

def make_huber_mat(k, residual):
    # caculate huber loss
    # print(residual.T)
    huber_weights = np.ones(8)
    for i in range(8):
        abs_err = residual[i,0]
        if abs_err > k:
            huber_weights[i] = k/abs_err
    # print("huberweights\n", huber_weights.T)
    return np.diag(huber_weights)

def huber_error(k, residual):
    error = 0
    for i in range(8):
        abs_err = residual[i,0]
        if abs_err > k:
            error += 2 * abs_err*k - k**2
        else:
            error += abs_err**2
    return error

# 6dof dimensional tag poses SE3
class MapBuilder:
    def __init__(self, camera_matrix, tag_side_lengths, map_type = "3d"):
        self.map_type = map_type

        self.regularizer = 1e9
        self.streak = 0

        self.huber_k = float(30)
        
        # assume camera matrix is rectified
        self.camera_matrix = np.array(camera_matrix)
        fx, fy, cx, cy = camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]
        self.camparams = np.array([[ fx, fy, cx, cy ]]).T
        self.inverse_pixel_cov = (1.0/10)**2
        self.tag_side_lengths = tag_side_lengths
        self.default_tag_side_length = tag_side_lengths["default"]
        self.corners_mats = []

        # linearization
        self.se3s_world_tag = []
        self.se3s_world_viewpoint = []

        self.txs_world_tag = []
        self.txs_world_viewpoint = []

        # detection factor data
        self.detection_jacobians = []
        self.detection_JtJs = []
        self.detection_rtJs = []
        self.detection_projections = []
        self.detection_residuals = []
        self.detection_errors = []

        # messages
        self.detection_to_tag_msgs = []
        self.detection_to_viewpoint_msgs = []

        # gbp state
        self.viewpoint_infos = []
        self.viewpoint_changes = []
        self.tag_infos = []
        self.tag_changes = []

        # list of (tag_idx, viewpoint_idx, tag_corners)
        self.detections = []

        self.viewpoint_id_to_idx = {}
        self.viewpoint_ids = []
        self.viewpoint_detections = [] # list of list
        self.tag_detections = [] # list of list
        self.tag_id_to_idx = {}
        self.tag_ids = []

        self.variable_priorities = MinHeap()

        # 1m above the surface, facing down onto the surface
        # height above the surface, facing down onto the surface
        h = self.default_tag_side_length * 10
        self.initial_viewpoint = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, h],
            [0,  0,  0, 1],
        ])

    def get_tag_side_length(self, tag_id):
        if tag_id in self.tag_side_lengths:
            tag_side_length = self.tag_side_lengths[tag_id]
        else:
            tag_side_length = self.default_tag_side_length
        return tag_side_length

    # def get_txs_world_viewpoint(self):
    #     return [ se3_exp(se3) for se3 in self.se3s_world_viewpoint ]

    def init_viewpoint(self, viewpoint_id, tx_world_viewpoint):
        assert viewpoint_id not in self.viewpoint_id_to_idx
        
        viewpoint_idx = len(self.viewpoint_ids)
        self.viewpoint_id_to_idx[viewpoint_id] = viewpoint_idx
        self.viewpoint_ids.append(viewpoint_id)

        se3_world_viewpoint = SE3_log(tx_world_viewpoint)

        self.se3s_world_viewpoint.append(se3_world_viewpoint)
        self.txs_world_viewpoint.append(tx_world_viewpoint)
        assert self.se3s_world_viewpoint[-1].shape == (6,1)

        self.viewpoint_infos.append(InfoState6())
        self.viewpoint_detections.append([])
        self.variable_priorities.upsert(('v', viewpoint_idx), 0)

        return viewpoint_idx

    def init_tag(self, tag_id, tx_world_tag):
        se3_world_tag = SE3_log(tx_world_tag)
        tag_idx = len(self.tag_ids)
        self.tag_id_to_idx[tag_id] = tag_idx
        self.tag_ids.append(tag_id)
        self.se3s_world_tag.append(se3_world_tag)
        self.txs_world_tag.append(tx_world_tag)
        self.tag_infos.append(InfoState6())
        self.variable_priorities.upsert(('t', tag_idx), 0)
        self.tag_detections.append([])
        tag_side_length = self.get_tag_side_length(tag_id)
        self.corners_mats.append(get_corners_mat(size=tag_side_length))

    def init_detection(self, viewpoint_id, tag_id, tag_corners):
        tag_idx = self.tag_id_to_idx[tag_id]
        viewpoint_idx = self.viewpoint_id_to_idx[viewpoint_id]
        dim_detection_factor_input = 12
        self.detections.append((tag_idx, viewpoint_idx, np.reshape(tag_corners, (8,1))))
        self.detection_jacobians.append(np.zeros(shape=(8,dim_detection_factor_input)))
        self.detection_projections.append(np.zeros(shape=(8,1)))
        self.detection_residuals.append(np.zeros(shape=(8,1)))
        self.detection_JtJs.append(np.zeros(shape=(dim_detection_factor_input, dim_detection_factor_input)))
        self.detection_rtJs.append(np.zeros(shape=(1,dim_detection_factor_input)))
        self.detection_to_viewpoint_msgs.append(InfoState6())
        self.detection_to_tag_msgs.append(InfoState6())
        self.detection_errors.append(float('inf'))
        self.tag_detections[tag_idx].append(len(self.detections)-1)
        self.viewpoint_detections[viewpoint_idx].append(len(self.detections)-1)

    def add_viewpoint(self, viewpoint_id, tags, initial_viewpoint = None, init_tags = None):
        if not self.tag_ids:
            # initialize the first tag
            first_tag_id = next(iter(tags.keys()))
            self.init_tag(first_tag_id, np.eye(4, dtype=float))
            
        if init_tags is None:
            init_tags = {}

        overlapping_tag_ids = list(
            self.tag_id_to_idx.keys() & tags.keys())

        if initial_viewpoint is None:
            if not overlapping_tag_ids:
                # there is no overlap with the existing map...
                # should we even allow this (?)
                if self.se3s_world_viewpoint:
                    # use the last viewpoint
                    print("No overlap with existing map!")
                    initial_viewpoint = se3_exp(self.se3s_world_viewpoint[-1])
                else:
                    # use the default init viewpoint
                    initial_viewpoint = self.initial_viewpoint
            else:
                # try to add this viewpoint into the scene
                # by registering against overlapping tag ids

                # get world coordinates of the overlapping tags
                tags_world_coords = []
                for tag_id in overlapping_tag_ids:
                    tag_idx = self.tag_id_to_idx[tag_id]
                    corners_mat = get_corners_mat(self.get_tag_side_length(tag_id))

                    # lift tag pose to SE3
                    tx_world_tag = se3_exp(self.se3s_world_tag[tag_idx])
                    tag_world_coords = (tx_world_tag @ corners_mat)[:3,:].T
                    tags_world_coords.append(tag_world_coords)

                # get pixel coordinates of the overlapping tags
                tags_img_coords = []
                for tag_id in overlapping_tag_ids:
                    tags_img_coords.append(np.array(tags[tag_id]).reshape(4,2))

                tags_world_coords = np.vstack(tags_world_coords)
                tags_img_coords = np.vstack(tags_img_coords)
                tx_camera_world = solvePnPWrapper(tags_world_coords, tags_img_coords, self.camera_matrix)
                initial_viewpoint = SE3_inv(tx_camera_world)

        # initialize the tag positions
        new_tag_ids = list((tags.keys() - self.tag_id_to_idx.keys()) - init_tags.keys())
        for tag_id in new_tag_ids:
            detection = tags[tag_id]
            detection = np.array(detection).reshape((4,2))
            corners_mat = np.array(get_corners_mat(self.get_tag_side_length(tag_id))[:3,:].T)
            tx_camera_tag = solvePnPWrapper(corners_mat, detection, self.camera_matrix)
            tx_world_tag = initial_viewpoint @ tx_camera_tag
            init_tags[tag_id] = tx_world_tag

        viewpoint_idx = self.init_viewpoint(viewpoint_id, initial_viewpoint)

        for tag_id, tag_corners in tags.items():
            if tag_id not in self.tag_id_to_idx:
                if tag_id in init_tags:
                    tx_world_tag = init_tags[tag_id]
                else:
                    tx_world_tag = np.eye(4, dtype=float)
                self.init_tag(tag_id, tx_world_tag)

            self.init_detection(viewpoint_id, tag_id, tag_corners)


    def update_viewpoint(self, viewpoint_idx):
        # [ ]_____( )
        # [ ]_____/
        # [ ]_____/
        # [ ]_____/

        viewpoint_info = self.viewpoint_infos[viewpoint_idx]
        prior_info = info6_from_gaussian(self.se3s_world_viewpoint[viewpoint_idx], 
                                         np.eye(6, dtype=float)*self.regularizer)

        new_info = prior_info + viewpoint_info
        new_se3 = np.linalg.solve(new_info.matrix, new_info.vector)

        change = np.linalg.norm(new_se3 - self.se3s_world_viewpoint[viewpoint_idx])
        self.variable_priorities.upsert(('v', viewpoint_idx), -change)

        assert new_se3.shape == (6,1)
        self.se3s_world_viewpoint[viewpoint_idx] = new_se3
        self.txs_world_viewpoint[viewpoint_idx] = se3_exp(new_se3)

    def get_tag_prior_info(self, tag_idx):
        prior_info = info6_from_gaussian(self.se3s_world_tag[tag_idx],
                                         np.eye(6, dtype=float)*self.regularizer)

        # add in a "factor" that adds cost when going away from z=0, xy rotation=0
        # || F tag ||^2 = || (1, 1, 0, 0, 0, 1) tag ||^2
        # tag F.t F tag

        dim_strength = 1e9
        
        if self.map_type == '2d':
            prior_info.matrix[:2,:2] += dim_strength
            prior_info.matrix[5,5] += dim_strength

        if self.map_type == '2.5d':
            prior_info.matrix[:2,:2] += dim_strength

        return prior_info


    def update_tag(self, tag_idx):
        # [ ]_____( )
        # [ ]_____/
        # [ ]_____/
        # [ ]_____/
        prior_info = self.get_tag_prior_info(tag_idx)
        tag_info = self.tag_infos[tag_idx]
        new_info = prior_info + tag_info
        new_se3 = np.linalg.solve(new_info.matrix, new_info.vector)
        assert new_se3.shape == (6,1)

        change = np.linalg.norm(new_se3 - self.se3s_world_tag[tag_idx])
        self.variable_priorities.upsert(('t', tag_idx), -change)
        
        self.se3s_world_tag[tag_idx] = new_se3
        self.txs_world_tag[tag_idx] = se3_exp(new_se3)

    def relinearize_detection(self, det_idx):
        detection = self.detections[det_idx]
        tag_idx, viewpoint_idx, tag_corners = detection
        se3_world_tag = self.se3s_world_tag[tag_idx]
        se3_world_viewpoint = self.se3s_world_viewpoint[viewpoint_idx]

        # can the view see the tag?
        tx_world_tag = self.txs_world_tag[tag_idx]
        tx_world_viewpoint = self.txs_world_viewpoint[viewpoint_idx]
        tx_viewpoint_tag = SE3_inv(tx_world_viewpoint) @ tx_world_tag

        tag_dir = tx_viewpoint_tag[:3,2].copy()
        to_tag = tx_viewpoint_tag[:3,3].copy()
        to_tag /= np.linalg.norm(to_tag)
        tag_dp = np.dot(tag_dir, to_tag)
        if tag_dp > 0:
            # camera is looking at the back of tag
            # express a factor saying that the
            self.detection_jacobians[det_idx].fill(0)
            self.detection_projections[det_idx].fill(0)
            self.detection_residuals[det_idx].fill(0)
            self.detection_JtJs[det_idx].fill(0)
            self.detection_rtJs[det_idx].fill(0)
            self.detection_errors[det_idx] = 1000
            return

        assert se3_world_viewpoint.shape == (6,1)

        image_corners, dimage_corners_dcamera, dimage_corners_dtag = project_points(self.camparams, se3_world_viewpoint, se3_world_tag, self.corners_mats[tag_idx])
        self.detection_jacobians[det_idx][:,:6] = dimage_corners_dcamera
        self.detection_jacobians[det_idx][:,6:] = dimage_corners_dtag
        self.detection_projections[det_idx] = image_corners
        residual = image_corners - tag_corners
        self.detection_residuals[det_idx] = residual

        # Numerical Derivative check
        # rng = np.random.default_rng(0)
        # direction_1 = rng.random((6,1))
        # direction_2 = rng.random((6,1))
        # epsilon = 1e-4
        # image_corners_d, _, _ = project_points(self.camparams, se3_world_viewpoint + epsilon*direction_1, se3_world_tag + epsilon*direction_2,
        #                                        self.corners_mats[tag_idx])
        # image_corners_deriv_d = (image_corners_d - image_corners)/epsilon
        # image_corners_deriv = dimage_corners_dcamera @ direction_1 + dimage_corners_dtag @ direction_2
        # print("Image corners deriv", image_corners_deriv)
        # print("Image corners deriv_d", image_corners_deriv_d)

        # caculate huber loss
        huber_k = self.huber_k
        huber_w = make_huber_mat(huber_k, residual)

        J = self.detection_jacobians[det_idx]
        self.detection_JtJs[det_idx] = self.inverse_pixel_cov * J.T @ huber_w @ J
        self.detection_rtJs[det_idx] = self.inverse_pixel_cov * residual.T @ huber_w @ J
        self.detection_errors[det_idx] = self.inverse_pixel_cov * huber_error(huber_k, residual)
        
    def relinearize(self):
        # print("Relinearizing")
        prev_error = self.get_total_detection_error()
        
        for i in range(len(self.detections)):
            self.relinearize_detection(i)
            
        curr_error = self.get_total_detection_error()
        if curr_error < prev_error:
            if self.streak > 10:
                self.regularizer *= 0.5                
            elif self.streak > 7:
                self.regularizer *= 0.7
            elif self.streak > 5:
                self.regularizer *= 0.9
            else:
                self.regularizer *= 0.99
        else:
            self.regularizer *= 25.0

        self.regularizer = max(self.regularizer, 1e-3)
        self.regularizer = min(self.regularizer, 1e1)

        return curr_error < prev_error

    def get_total_detection_error(self):
        return sum(self.detection_errors)

    def get_avg_detection_error(self):
        return self.get_total_detection_error()/len(self.detection_errors)

    def reset_tag(self, tag_id, tx_world_tag):
        tag_idx = self.tag_id_to_idx[tag_id]
        self.se3s_world_tag[tag_idx] = SE3_log(tx_world_tag)
        self.txs_world_tag[tag_idx] = tx_world_tag

        for detection_idx in self.tag_detections[tag_idx]:
            # reset the message to this tag
            self.detection_to_tag_msgs[detection_idx] = InfoState6()
            self.relinearize_detection(detection_idx)

        self.variable_priorities.upsert(tag_idx, 0)
        
        # update the tag info ?
        # clear out all the messages to this tag ?

    def prioritized_update(self):
        if not self.detections:
            return

        detections_needing_relin = set()

        (var_type, var_idx), priority = self.variable_priorities.pop()
        if var_type == 't':
            viewpoints_messaged = set()
            for detection_idx in self.tag_detections[var_idx]:
                tag_idx, viewpoint_idx, _ = self.detections[detection_idx]
                self.send_detection_to_viewpoint_msg(detection_idx)
                viewpoints_messaged.add(viewpoint_idx)
                detections_needing_relin.add(detection_idx)

            for viewpoint_idx in viewpoints_messaged:
                self.update_viewpoint(viewpoint_idx)
        else:
            tags_messaged = set()
            for detection_idx in self.viewpoint_detections[var_idx]:
                tag_idx, viewpoint_idx, _ = self.detections[detection_idx]                                        
                self.send_detection_to_tag_msg(detection_idx)
                tags_messaged.add(tag_idx)
                detections_needing_relin.add(detection_idx)

            for tag_idx in tags_messaged:
                self.update_tag(tag_idx)

        for det_idx in detections_needing_relin:
            self.relinearize_detection(det_idx)

    def update(self):
        # copy linearization point
        # se3s_world_viewpoint_backup = [tx.copy() for tx in self.se3s_world_viewpoint]
        # se3s_world_tag_backup = [tx.copy() for tx in self.se3s_world_tag]
        # txs_world_viewpoint_backup = [tx.copy() for tx in self.txs_world_viewpoint]
        # txs_world_tag_backup = [tx.copy() for tx in self.txs_world_tag]

        for viewpoint_idx in range(len(self.viewpoint_infos)):
            self.update_viewpoint(viewpoint_idx)

        for tag_idx in range(len(self.tag_infos)):
            self.update_tag(tag_idx)
            # prior_info = info6_from_gaussian(self.se3s_world_tag[tag_idx], 
            #                                  np.eye(6, dtype=float)*self.regularizer)

            # new_info = prior_info + tag_info
            # new_se3 = np.linalg.solve(new_info.matrix, new_info.vector)
            # assert new_se3.shape == (6,1)
            # self.se3s_world_tag[tag_idx] = new_se3
            # self.txs_world_tag[tag_idx] = se3_exp(new_se3)

        if not self.relinearize():
            pass
            # # no improvement, restore the previous linearization point
            # self.streak = 0
            # self.se3s_world_viewpoint = se3s_world_viewpoint_backup
            # self.se3s_world_tag = se3s_world_tag_backup
            # self.txs_world_tag = txs_world_tag_backup
            # self.txs_world_viewpoint = txs_world_viewpoint_backup
            # self.relinearize()
            # return False

        # print("improvement. regularizer is now", self.regularizer)
        self.streak += 1
        return True

    def send_detection_to_viewpoint_msgs(self):
        # print("Sending detection to viewpoint msgs")
        for detection_idx, (tag_idx, viewpoint_idx, _) in enumerate(self.detections):
            self.send_detection_to_viewpoint_msg(detection_idx)

    def send_detection_to_tag_msgs(self):
        # print("Sending detection to tag msgs")
        for detection_idx, (tag_idx, viewpoint_idx, _) in enumerate(self.detections):
            self.send_detection_to_tag_msg(detection_idx)

    def send_detection_to_viewpoint_msg(self, detection_idx):
        tag_idx, viewpoint_idx, _ = self.detections[detection_idx]

        # detection to camera message
        #            __[ detectionA ]________
        #       ...  __[ detectionB ]________\
        #            __[ detectionC ]________\
        #                                    \
        # ( view )_____[ detection  ]______( tag )____[tag prior]
        #    \_________[ detectionE ]__
        #    \_________[ detectionF ]__  ...
        #    \_________[ detectionG ]__

        # get the message from the tag to the viewpoint
        # the sum of all the messages to the except for the one sent by this view
        tag_prior = info6_from_gaussian(self.se3s_world_tag[tag_idx], 1.0/self.regularizer * np.eye(6, dtype=float))
        tag_info = self.tag_infos[tag_idx] - self.detection_to_tag_msgs[detection_idx] + tag_prior

        # marginalize out the tag
        # and send into the view
        # 
        # joint distribution = exp(-det_cost(tag, view, detection))*exp(-½ tag.t * tag_info_mat * tag + tag_info_vec.t * tag)
        # or in -logprob terms,
        # 
        # -logprob =  ½ tag.t Λt tag - ηt.t tag + || Jdet ⎡Δview⎤ + det_residual||²
        #                                                 ⎣ Δtag⎦
        #
        #          =  ½ tag.t Λt tag - ηt.t tag + || Jdet ⎡view - linview⎤ + det_residual||²
        #                                                 ⎣tag  - lintag ⎦
        #                
        #          =  ½ tag.t Λt tag - ηt.t tag + || Jdet ⎡view - linview⎤ + det_residual||²
        #                                                 ⎣tag  - lintag ⎦
        #                
        #          = ½ tag.t Λt tag - ηt.t tag + 
        #            ⎡view - linview⎤.t Jdet.t Jdet ⎡view - linview⎤ + 2 det_residual.t Jdet ⎡view - linview⎤
        #            ⎣tag -  lintag ⎦               ⎣tag -  lintag ⎦                         ⎣tag  - lintag ⎦
        #                                       
        #          = ½ tag.t Λt tag - ηt.t tag + 
        #            ⎡view⎤.t Jdet.t Jdet ⎡view⎤ + 2 det_residual.t Jdet ⎡view ⎤ -  2 ⎡linview⎤.t Jdet.t Jdet ⎡view⎤
        #            ⎣tag ⎦               ⎣tag ⎦                         ⎣tag  ⎦      ⎣lintag ⎦               ⎣tag ⎦
        #            + constant
        # 
        #          = ½ tag.t Λt tag - ηt.t tag + 
        #            ⎡view⎤.t Jdet.t Jdet ⎡view⎤ - (-2 det_residual.t Jdet + 2 ⎡linview⎤.t Jdet.t Jdet) ⎡view⎤
        #            ⎣tag ⎦               ⎣tag ⎦                               ⎣lintag ⎦                ⎣tag ⎦
        #            + constant
                                           
        # information space marginalization
        # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf page 6
        #
        # marginalize out the tag component
        # Λc' = Λcc - Λct Λtt⁻¹ Λtc
        # ηc' = ηc - Λct Λtt⁻¹ ηt

        linpoint = np.empty((12, 1))
        linpoint[:6,:] = self.se3s_world_viewpoint[viewpoint_idx]
        linpoint[6:,:] = self.se3s_world_tag[tag_idx]

        JtJ = self.detection_JtJs[detection_idx]
        rtJ = self.detection_rtJs[detection_idx]

        total_info_matrix = 2*JtJ
        total_info_matrix[6:,6:] += tag_info.matrix
        total_info_vector = (-2*rtJ + 2*linpoint.T @ JtJ).T

        total_info_vector[6:,:] += tag_info.vector

        lambda_cc = total_info_matrix[:6,:6]
        lambda_ct = total_info_matrix[:6,6:]
        lambda_tt = total_info_matrix[6:,6:]

        nu_t = total_info_vector[6:,:]
        nu_c = total_info_vector[:6,:]

        matrix_msg = lambda_cc - lambda_ct @ (np.linalg.solve(lambda_tt, lambda_ct.T))
        #                                        lambda_tt.inverse() @ lambda_tc
        vector_msg = nu_c - lambda_ct @ np.linalg.solve(lambda_tt, nu_t)
        #                                  lambda_tt.inverse() @ nu_t
        assert nu_c.shape == (6,1)
        assert vector_msg.shape == (6,1)

        msg = InfoState6(vector_msg, matrix_msg)

        self.viewpoint_infos[viewpoint_idx] -= self.detection_to_viewpoint_msgs[detection_idx] # undo the previous message from this det
        self.viewpoint_infos[viewpoint_idx] += msg # add on the current message from this det
        assert self.viewpoint_infos[viewpoint_idx].vector.shape == (6,1)
        self.detection_to_viewpoint_msgs[detection_idx] = msg


    def send_detection_to_tag_msg(self, detection_idx):
        tag_idx, viewpoint_idx, _ = self.detections[detection_idx]

        # detection to camera message
        #              [ detectionA ]________
        #              [ detectionB ]________\
        #              [ detectionC ]________\
        #                                    \
        # ( tag )______[ detection  ]______( view )____[view prior]
        #    \_________[ detectionE ]
        #    \_________[ detectionF ]
        #    \_________[ detectionG ]

        # get the message from the viewpoint to the tag to the
        viewpoint_prior = info6_from_gaussian(self.se3s_world_viewpoint[viewpoint_idx], 1.0/self.regularizer*np.eye(6, dtype=float))
        viewpoint_info = self.viewpoint_infos[viewpoint_idx] - self.detection_to_viewpoint_msgs[detection_idx] + viewpoint_prior

        # marginalize out the viewpoint
        # and send into the tag
        # -logprob =  ½ view.t Λc view - ηc.t view + || Jdet ⎡Δview⎤ + det_residual||²
        #                                                 ⎣ Δtag⎦
        #
        #          =  ½ view.t Λc view - ηc.t view + || Jdet ⎡view - linview⎤ + det_residual||²
        #                                                 ⎣tag  - lintag ⎦
        #                
        #          =  ½ view.t Λc view - ηc.t view + || Jdet ⎡view - linview⎤ + det_residual||²
        #                                                 ⎣tag  - lintag ⎦
        #                
        #          = ½ view.t Λc view - ηc.t view + 
        #            ⎡view - linview⎤.t Jdet.t Jdet ⎡view - linview⎤ + 2 det_residual.t Jdet ⎡view - linview⎤
        #            ⎣tag -  lintag ⎦               ⎣tag -  lintag ⎦                         ⎣tag  - lintag ⎦
        #                                       
        #          = ½ view.t Λc view - ηc.t view + 
        #            ⎡view⎤.t Jdet.t Jdet ⎡view⎤ + 2 det_residual.t Jdet ⎡view ⎤ -  2 ⎡linview⎤.t Jdet.t Jdet ⎡view⎤
        #            ⎣tag ⎦               ⎣tag ⎦                         ⎣tag  ⎦      ⎣lintag ⎦               ⎣tag ⎦
        #            + constant
        # 
        #          = ½ view.t Λc view - ηc.t view + 
        #            ⎡view⎤.t Jdet.t Jdet ⎡view⎤ - (-2det_residual.t Jdet + 2 ⎡linview⎤.t Jdet.t Jdet) ⎡view⎤
        #            ⎣tag ⎦               ⎣tag ⎦                              ⎣lintag ⎦                ⎣tag ⎦
        #            + constant
        #
        # 
        # information space marginalization
        # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf page 6
        #
        # marginalize out the tag component
        # Λt' = Λtt - Λtc Λcc⁻¹ Λct
        # ηt' = ηt - Λtc Λcc⁻¹ ηc

        linpoint = np.empty((12, 1))
        linpoint[:6,:] = self.se3s_world_viewpoint[viewpoint_idx]
        linpoint[6:,:] = self.se3s_world_tag[tag_idx]

        JtJ = self.detection_JtJs[detection_idx]
        rtJ = self.detection_rtJs[detection_idx]

        total_info_matrix = 2*JtJ
        total_info_matrix[:6,:6] += viewpoint_info.matrix

        total_info_vector = (-2*rtJ + 2*linpoint.T @ JtJ).T
        total_info_vector[:6,:] += viewpoint_info.vector

        lambda_cc = total_info_matrix[:6,:6]
        lambda_ct = total_info_matrix[:6,6:]
        lambda_tt = total_info_matrix[6:,6:]

        nu_t = total_info_vector[6:,:]
        nu_c = total_info_vector[:6,:]

        matrix_msg = lambda_tt - lambda_ct.T @ (np.linalg.solve(lambda_cc, lambda_ct))
        #                                        lambda_cc.inverse() @ lambda_ct
        vector_msg = nu_t - lambda_ct.T @ np.linalg.solve(lambda_cc, nu_c)
        #                                     lambda_cc.inverse() @ nu_c

        msg = InfoState6(vector_msg, matrix_msg)
        self.tag_infos[tag_idx] -= self.detection_to_tag_msgs[detection_idx] # undo the previous message from this det
        self.detection_to_tag_msgs[detection_idx] = msg
        self.tag_infos[tag_idx] += msg # add on the current message from this det
