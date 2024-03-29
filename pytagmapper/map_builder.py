import numpy as np
import math
from pytagmapper.geometry import *
from pytagmapper.info_state import *
from pytagmapper.project import project, get_corners_mat
from pytagmapper.heuristics import *
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
    def __init__(self, camera_matrix, tag_side_lengths, map_type = "2d"):
        self.map_type = map_type

        if map_type == "3d":
            self.tag_dof = 6 # wx wy wz x y z
            self.tag_info_cls = InfoState6
            self.tx_world_tag_dim = 4 # 4x4 pose matrix
        elif map_type == "2.5d":
            self.tag_dof = 4 # wz x y z
            self.tag_info_cls = InfoState4
            self.tx_world_tag_dim = 4 # 4x4 pose matrix
        elif map_type == "2d":
            self.tag_dof = 3 # wz x y 
            self.tag_info_cls = InfoState3
            self.tx_world_tag_dim = 3 # 3x3 pose matrix
        else:
            raise RuntimeError("Unsupported map type", map_type)
        
        self.regularizer = 1e9
        self.streak = 0

        self.huber_k = float(30)
        
        self.camera_matrix = np.array(camera_matrix)
        self.inverse_pixel_cov = (1.0/10)**2
        self.tag_side_lengths = tag_side_lengths
        self.default_tag_side_length = tag_side_lengths["default"]
        self.corners_mats = []

        # linearization
        self.txs_world_viewpoint = []
        self.txs_world_tag = []

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
        self.tag_infos = []

        # list of (tag_idx, viewpoint_idx, tag_corners)
        self.detections = []

        self.viewpoint_id_to_idx = {}
        self.viewpoint_ids = []
        self.viewpoint_detections = [] # list of (start, finish) tuple
        self.tag_detections = [] # list of list
        self.tag_id_to_idx = {}
        self.tag_ids = []

        # 1m above the surface, facing down onto the surface
        # height above the surface, facing down onto the surface
        h = self.default_tag_side_length * 10
        self.init_viewpoint = np.array([
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

    def check_dims(self):
        for tx in self.txs_world_tag:
            if tx.shape != (self.tx_world_tag_dim, self.tx_world_tag_dim):
                raise RuntimeError("Bad dimension", tx.shape)

    def add_viewpoint(self, viewpoint_id, tags, init_viewpoint = None, init_tags = None):
        if init_tags is None:
            init_tags = {}

        self.viewpoint_id_to_idx[viewpoint_id] = len(self.viewpoint_ids)
        self.viewpoint_ids.append(viewpoint_id)

        overlapping_tag_ids = list(
            self.tag_id_to_idx.keys() & tags.keys())

        if init_viewpoint is None:
            if not overlapping_tag_ids:
                # there is no overlap with the existing map...
                # should we even allow this (?)
                if self.txs_world_viewpoint:
                    # use the last viewpoint
                    print("No overlap with existing map!")
                    init_viewpoint = self.txs_world_viewpoint[-1]
                else:
                    # use the default init viewpoint
                    init_viewpoint = self.init_viewpoint
            else:
                # try to add this viewpoint into the scene
                # by registering against overlapping tag ids

                # print("Initializing viewpoint")

                # get world coordinates of the overlapping tags
                tags_world_coords = []
                for tag_id in overlapping_tag_ids:
                    tag_idx = self.tag_id_to_idx[tag_id]
                    corners_mat = get_corners_mat(self.get_tag_side_length(tag_id))

                    # lift tag pose to SE3
                    if self.map_type == '2d':
                        tx_world_tag = SE2_to_SE3(self.txs_world_tag[tag_idx])
                    else:
                        tx_world_tag = self.txs_world_tag[tag_idx]

                    tag_world_coords = (tx_world_tag @ corners_mat)[:3,:].T
                    tags_world_coords.append(tag_world_coords)

                # get pixel coordinates of the overlapping tags
                tags_img_coords = []
                for tag_id in overlapping_tag_ids:
                    tags_img_coords.append(np.array(tags[tag_id]).reshape(4,2))

                tags_world_coords = np.vstack(tags_world_coords)
                tags_img_coords = np.vstack(tags_img_coords)
                tx_camera_world = solvePnPWrapper(tags_world_coords, tags_img_coords, self.camera_matrix)
                init_viewpoint = SE3_inv(tx_camera_world)

        # initialize the tag positions
        new_tag_ids = list((tags.keys() - self.tag_id_to_idx.keys()) - init_tags.keys())
        for tag_id in new_tag_ids:
            # print("initializing tag", tag_id)
            detection = tags[tag_id]
            detection = np.array(detection).reshape((4,2))
            corners_mat = np.array(get_corners_mat(self.get_tag_side_length(tag_id))[:3,:].T)
            tx_camera_tag = solvePnPWrapper(corners_mat, detection, self.camera_matrix)
            tx_world_tag_3d = init_viewpoint @ tx_camera_tag

            if self.map_type == '2d':
                tx_world_tag = SE3_to_SE2(tx_world_tag_3d)
            elif self.map_type == '2.5d':
                tx_world_tag_2d = SE3_to_SE2(tx_world_tag_3d)
                tx_world_tag = SE2_to_SE3(tx_world_tag_2d)
                tx_world_tag[2,3] = tx_world_tag_3d[2,3] # fix z component
            elif self.map_type == '3d':
                tx_world_tag = tx_world_tag_3d
            else:
                raise RuntimeError("Unsupported map type", self.map_type)
            
            init_tags[tag_id] = tx_world_tag

        self.txs_world_viewpoint.append(init_viewpoint)
        self.viewpoint_infos.append(InfoState6())
        viewpoint_idx = self.viewpoint_id_to_idx[viewpoint_id]
        viewpoint_detections_start = len(self.detections)
        for tag_id, tag_corners in tags.items():
            if tag_id not in self.tag_id_to_idx:
                self.tag_id_to_idx[tag_id] = len(self.tag_ids)
                self.tag_ids.append(tag_id)
                if tag_id in init_tags:
                    self.txs_world_tag.append(init_tags[tag_id])
                else:
                    self.txs_world_tag.append(np.eye(self.tx_world_tag_dim))
                self.tag_infos.append(self.tag_info_cls())
                self.tag_detections.append([])
                tag_side_length = self.get_tag_side_length(tag_id)
                self.corners_mats.append(get_corners_mat(size=tag_side_length))

            tag_idx = self.tag_id_to_idx[tag_id]

            dim_detection_factor_input = 6 + self.tag_dof
            self.detections.append((tag_idx, viewpoint_idx, np.reshape(tag_corners, (8,1))))
            self.detection_jacobians.append(np.zeros(shape=(8,dim_detection_factor_input)))
            self.detection_projections.append(np.zeros(shape=(8,1)))
            self.detection_residuals.append(np.zeros(shape=(8,1)))
            self.detection_JtJs.append(np.zeros(shape=(dim_detection_factor_input, dim_detection_factor_input)))
            self.detection_rtJs.append(np.zeros(shape=(1,dim_detection_factor_input)))
            self.detection_to_viewpoint_msgs.append(InfoState6())
            self.detection_to_tag_msgs.append(self.tag_info_cls())
            self.detection_errors.append(float('inf'))
            self.tag_detections[tag_idx].append(len(self.detections)-1)

        viewpoint_detections_end = len(self.detections)
        self.viewpoint_detections.append((viewpoint_detections_start, viewpoint_detections_end))

    def update_viewpoint(self, viewpoint_idx):
        # [ ]_____( )
        # [ ]_____/
        # [ ]_____/
        # [ ]_____/

        viewpoint_info = self.viewpoint_infos[viewpoint_idx]        
        delta = np.linalg.solve(viewpoint_info.matrix, viewpoint_info.vector)
        self.txs_world_viewpoint[viewpoint_idx] = self.txs_world_viewpoint[viewpoint_idx] @ se3_exp(delta)
        fix_SE3(self.txs_world_viewpoint[viewpoint_idx])
        viewpoint_info.clear()
        viewpoint_info.matrix = self.regularizer * np.eye(6)

        # apply update
        # relinearize all detections involved with this viewpoint
        # send updates to this viewpoint
        for det_idx in range(*self.viewpoint_detections[viewpoint_idx]):
            self.relinearize_detection(det_idx)

        # send updates to this viewpoint from all detections
        for det_idx in range(*self.viewpoint_detections[viewpoint_idx]):
            self.detection_to_viewpoint_msgs[det_idx].clear()                
            self.send_detection_to_viewpoint_msg(det_idx)

        # send updates from this viewpoint to all detections
        for det_idx in range(*self.viewpoint_detections[viewpoint_idx]):
            self.send_detection_to_tag_msg(det_idx)

    def update_tag(self, tag_idx):
        # [ ]_____( )
        # [ ]_____/
        # [ ]_____/
        # [ ]_____/

        tag_info = self.tag_infos[tag_idx]

        delta = np.linalg.solve(tag_info.matrix, tag_info.vector)
        if self.map_type == "2d":
            self.txs_world_tag[tag_idx] = self.txs_world_tag[tag_idx] @ se2_exp(delta)
        elif self.map_type == "2.5d":
            # haven't implemented exp for SE2xR, so just lift to SE3
            se3_delta = np.zeros((6,1))
            se3_delta[2:,:] = delta # [0,0,wz,x,y,z]
            self.txs_world_tag[tag_idx] = self.txs_world_tag[tag_idx] @ se3_exp(se3_delta)
        elif self.map_type == "3d":
            self.txs_world_tag[tag_idx] = self.txs_world_tag[tag_idx] @ se3_exp(delta)
        else:
            raise RuntimeError("Unsupported map type", self.map_type)
        
        tag_info.clear()
        tag_info.matrix = self.regularizer * np.eye(self.tag_dof)

        # apply update
        # relinearize all detections involved with this tag
        for det_idx in self.tag_detections[tag_idx]:
            self.relinearize_detection(det_idx)

        # send updates to this tag from all detections
        for det_idx in self.tag_detections[tag_idx]:
            self.detection_to_tag_msgs[det_idx].clear()
            self.send_detection_to_tag_msg(det_idx)

        # send updates from this tag to all detections
        for det_idx in self.tag_detections[tag_idx]:
            self.send_detection_to_viewpoint_msg(det_idx)

    def relinearize_detection(self, det_idx):
        detection = self.detections[det_idx]
        tag_idx, viewpoint_idx, tag_corners = detection
        tx_world_tag = self.txs_world_tag[tag_idx]
        if self.map_type == "2d":
            tx_world_tag = SE2_to_SE3(tx_world_tag)

        tx_world_viewpoint = self.txs_world_viewpoint[viewpoint_idx]
        tx_viewpoint_tag = SE3_inv(tx_world_viewpoint) @ tx_world_tag
        image_corners, dimage_corners_dcamera, dimage_corners_dtag = project(self.camera_matrix, tx_viewpoint_tag, self.corners_mats[tag_idx])
        self.detection_jacobians[det_idx][:,:6] = dimage_corners_dcamera
        if self.map_type == "2d":
            self.detection_jacobians[det_idx][:,6+0] = dimage_corners_dtag[:,2] # wz
            self.detection_jacobians[det_idx][:,6+1] = dimage_corners_dtag[:,3] # dx
            self.detection_jacobians[det_idx][:,6+2] = dimage_corners_dtag[:,4] # dy
        elif self.map_type == "2.5d":
            self.detection_jacobians[det_idx][:,6+0] = dimage_corners_dtag[:,2] # wz
            self.detection_jacobians[det_idx][:,6+1] = dimage_corners_dtag[:,3] # dx
            self.detection_jacobians[det_idx][:,6+2] = dimage_corners_dtag[:,4] # dy
            self.detection_jacobians[det_idx][:,6+3] = dimage_corners_dtag[:,5] # dz
        elif self.map_type == "3d":
            self.detection_jacobians[det_idx][:,6:] = dimage_corners_dtag
        else:
            raise RuntimeError("Unsupported map type", self.map_type)

        self.detection_projections[det_idx] = image_corners
        residual = image_corners - tag_corners
        self.detection_residuals[det_idx] = residual

        # caculate huber loss
        huber_k = self.huber_k
        huber_w = make_huber_mat(huber_k, residual)

        J = self.detection_jacobians[det_idx]
        self.detection_JtJs[det_idx] = self.inverse_pixel_cov * J.T @ huber_w @ J
        self.detection_rtJs[det_idx] = self.inverse_pixel_cov * residual.T @ huber_w @ J
        self.detection_errors[det_idx] = self.inverse_pixel_cov * huber_error(huber_k, residual)

        
    def relinearize(self):
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
        self.regularizer = min(self.regularizer, 1e6)

        # clear all messages and states
        # since these are not valid for the new linearization point
        for msg in self.detection_to_tag_msgs:
            msg.clear()
        for msg in self.detection_to_viewpoint_msgs:
            msg.clear()

        for info in self.viewpoint_infos:
            info.clear()
            info.matrix = self.regularizer * np.eye(6)
        for info in self.tag_infos:
            info.clear()
            info.matrix = self.regularizer * np.eye(self.tag_dof)

        return curr_error < prev_error

    def get_total_detection_error(self):
        return sum(self.detection_errors)

    def get_avg_detection_error(self):
        return self.get_total_detection_error()/len(self.detection_errors)

    def update(self):
        # copy linearization point
        txs_world_viewpoint_backup = [tx.copy() for tx in self.txs_world_viewpoint]
        txs_world_tag_backup = [tx.copy() for tx in self.txs_world_tag]

        for viewpoint_idx, viewpoint_info in enumerate(self.viewpoint_infos):
            delta = np.linalg.solve(viewpoint_info.matrix, viewpoint_info.vector)
            self.txs_world_viewpoint[viewpoint_idx] = self.txs_world_viewpoint[viewpoint_idx] @ se3_exp(delta)
            fix_SE3(self.txs_world_viewpoint[viewpoint_idx])

        for tag_idx, tag_info in enumerate(self.tag_infos):
            delta = np.linalg.solve(tag_info.matrix, tag_info.vector)
            if self.map_type == "2d":
                self.txs_world_tag[tag_idx] = self.txs_world_tag[tag_idx] @ se2_exp(delta)
            elif self.map_type == "2.5d":
                # haven't implemented exp for SE2xR, so just lift to SE3
                se3_delta = np.zeros((6,1))
                se3_delta[2:,:] = delta # [0,0,wz,x,y,z]
                self.txs_world_tag[tag_idx] = self.txs_world_tag[tag_idx] @ se3_exp(se3_delta)
            elif self.map_type == "3d":
                self.txs_world_tag[tag_idx] = self.txs_world_tag[tag_idx] @ se3_exp(delta)
            else:
                raise RuntimeError("Unsupported map type", self.map_type)

        # recenter the map around tag0            
        if self.tx_world_tag_dim == 3:
            tx_world_tag0 = self.txs_world_tag[0]
            tx_tag0_world = SE2_inv(tx_world_tag0)
            for i, tx_world_tag in enumerate(self.txs_world_tag):
                self.txs_world_tag[i] = tx_tag0_world @ self.txs_world_tag[i]
                fix_SE2(self.txs_world_tag[i])
            tx_tag0_world = SE2_to_SE3(tx_tag0_world) # promote to SE2->SE3
            for i, tx_world_viewpoint in enumerate(self.txs_world_viewpoint):
                self.txs_world_viewpoint[i] = tx_tag0_world @ tx_world_viewpoint
                fix_SE3(self.txs_world_viewpoint[i])

        elif self.tx_world_tag_dim == 4:
            tx_world_tag0 = self.txs_world_tag[0]
            tx_tag0_world = SE3_inv(tx_world_tag0)
            for i, tx_world_tag in enumerate(self.txs_world_tag):
                self.txs_world_tag[i] = tx_tag0_world @ self.txs_world_tag[i]
                fix_SE3(self.txs_world_tag[i])
            for i, tx_world_viewpoint in enumerate(self.txs_world_viewpoint):
                self.txs_world_viewpoint[i] = tx_tag0_world @ tx_world_viewpoint
                fix_SE3(self.txs_world_viewpoint[i])
        else:
            raise RuntimeError("Unexpected tag pose dimention", self.tx_world_tag_dim)

        if not self.relinearize():
            # no improvement, restore the previous linearization point
            self.streak = 0
            self.txs_world_viewpoint = txs_world_viewpoint_backup
            self.txs_world_tag = txs_world_tag_backup
            self.relinearize()
            return False

        # print("improvement. regularizer is now", self.regularizer)
        self.streak += 1
        return True

    def send_detection_to_viewpoint_msgs(self):
        for detection_idx, (tag_idx, viewpoint_idx, _) in enumerate(self.detections):
            self.send_detection_to_viewpoint_msg(detection_idx)

    def send_detection_to_tag_msgs(self):
        for detection_idx, (tag_idx, viewpoint_idx, _) in enumerate(self.detections):
            self.send_detection_to_tag_msg(detection_idx)

    def send_detection_to_viewpoint_msg(self, detection_idx):
        tag_idx, viewpoint_idx, _ = self.detections[detection_idx]

        # detection to camera message
        #            __[ detectionA ]________
        #       ...  __[ detectionB ]________\
        #            __[ detectionC ]________\
        #                                    \
        # ( view )_____[ detection  ]______( tag )
        #    \_________[ detectionE ]__
        #    \_________[ detectionF ]__  ...
        #    \_________[ detectionG ]__

        # get the message from the tag to the viewpoint
        tag_info = self.tag_infos[tag_idx] - self.detection_to_tag_msgs[detection_idx]

        # print("tag to viewpoint msg")
        # print(tag_info.vector.T)
        # print(tag_info.matrix)

        # marginalize out the tag
        # and send into the view
        # 
        # cost = det(tag, view, detection) + tag
        # cost =  ½ Δtag.t Λt Δtag - ηt.t Δtag + || Jdet ⎡Δview⎤ + det_residual||² +
        #                                                ⎣ Δtag⎦
        #                
        #      =  ½ Δtag.t Λt Δtag - ηt.t Δtag +
        #         [Δview.t, Δtag.t ] Jdet.t Jdet ⎡Δview⎤ + 2 det_residual.t Jdet ⎡Δview⎤
        #                                        ⎣Δtag ⎦                         ⎣Δtag ⎦
        #                                       
        #      = ½ Δtag.t Λt Δtag - ηt.t Δtag +
        #        ½ [Δview.t, Δtag.t ] 2 Jdet.t Jdet ⎡Δview⎤ + 2 det_residual.t Jdet ⎡Δview⎤
        #                                           ⎣Δtag ⎦                         ⎣Δtag ⎦
        #                                            
        # information space marginalization
        # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf page 6
        #
        # marginalize out the tag component
        # Λc' = Λcc - Λct Λtt⁻¹ Λtc
        # ηc' = ηc - Λct Λtt⁻¹ ηt

        JtJ = self.detection_JtJs[detection_idx]
        rtJ = self.detection_rtJs[detection_idx]

        total_info_matrix = 2*JtJ
        total_info_matrix[6:,6:] += tag_info.matrix
        total_info_vector = -2*rtJ.T
        total_info_vector[6:,:] += tag_info.vector

        lambda_cc = total_info_matrix[:6,:6]
        lambda_ct = total_info_matrix[:6,6:]
        lambda_tt = total_info_matrix[6:,6:]

        nu_t = total_info_vector[6:,:]
        nu_c = total_info_vector[:6,:]

        # print("DET TO VAR MSG==========")
        # print("JtJ", JtJ)
        # print("rtJ", rtJ)
        # print("lambda_cc", lambda_cc)
        # print("lambda_tt", lambda_tt)
        # print("lambda_ct", lambda_ct)
        # print("nu_c", nu_c)
        # print("nu_t", nu_t)

        matrix_msg = lambda_cc - lambda_ct @ (np.linalg.solve(lambda_tt, lambda_ct.T))
        #                                        lambda_tt.inverse() @ lambda_tc
        vector_msg = nu_c - lambda_ct @ np.linalg.solve(lambda_tt, nu_t)
        #                                  lambda_tt.inverse() @ nu_t

        msg = InfoState6(vector_msg, matrix_msg)
        self.viewpoint_infos[viewpoint_idx] -= self.detection_to_viewpoint_msgs[detection_idx] # undo the previous message from this det
        self.viewpoint_infos[viewpoint_idx] += msg # add on the current message from this det
        self.detection_to_viewpoint_msgs[detection_idx] = msg


    def send_detection_to_tag_msg(self, detection_idx):
        tag_idx, viewpoint_idx, _ = self.detections[detection_idx]

        # detection to camera message
        #              [ detectionA ]________
        #              [ detectionB ]________\
        #              [ detectionC ]________\
        #                                    \
        # ( view )_____[ detection  ]______( tag )
        #    \_________[ detectionE ]
        #    \_________[ detectionF ]
        #    \_________[ detectionG ]

        # get the message from the tag to the viewpoint
        viewpoint_info = self.viewpoint_infos[viewpoint_idx] - self.detection_to_viewpoint_msgs[detection_idx]

        # marginalize out the viewpoint
        # and send into the tag
        # 
        # cost = det(tag, view, detection) + tag
        # cost =  ½ Δview.t Λv Δview - ηv.t Δview + || Jdet ⎡Δview⎤ + det_residual||² +
        #                                                   ⎣ Δtag⎦
        #                
        #      = ½ Δtag.t Λt Δtag - ηv.t Δtag +
        #        ½ [Δview.t, Δtag.t ] 2 Jdet.t Jdet ⎡Δview⎤ + 2 det_residual.t Jdet ⎡Δview⎤
        #                                           ⎣Δtag ⎦                         ⎣Δtag ⎦
        #                                            
        # information space marginalization
        # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf page 6
        #
        # marginalize out the tag component
        # Λt' = Λtt - Λtc Λcc⁻¹ Λct
        # ηt' = ηt - Λtc Λcc⁻¹ ηc

        JtJ = self.detection_JtJs[detection_idx]
        rtJ = self.detection_rtJs[detection_idx]

        total_info_matrix = 2*JtJ
        total_info_matrix[:6,:6] += viewpoint_info.matrix

        total_info_vector = -2*rtJ.T
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

        msg = self.tag_info_cls(vector_msg, matrix_msg)
        self.tag_infos[tag_idx] -= self.detection_to_tag_msgs[detection_idx] # undo the previous message from this det
        self.detection_to_tag_msgs[detection_idx] = msg
        self.tag_infos[tag_idx] += msg # add on the current message from this det
