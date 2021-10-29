import numpy as np
import math
from geometry import *
from info_state import *
from project import project, get_corners_mat
from heuristics import *

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
class MapBuilder3d:
    DIM_VIEWPOINT_POSE = 6
    DIM_TAG_POSE = 6
    
    def __init__(self, camera_matrix, tag_side_length):
        self.regularizer = 1e9
        self.streak = 0
        
        self.camera_matrix = np.array(camera_matrix)
        self.inverse_pixel_cov = (1.0/10)**2
        self.tag_side_length = tag_side_length
        self.corners_mat = get_corners_mat(size=tag_side_length)

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
        self.viewpoint_detections = []
        self.tag_id_to_idx = {}
        self.tag_ids = []

        # 1m above the surface, facing down onto the surface
        # height above the surface, facing down onto the surface
        h = self.tag_side_length * 10
        self.init_viewpoint = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, h],
            [0,  0,  0, 1],
        ])

    def add_viewpoint(self, viewpoint_id, tags, init_viewpoint = None):
        self.viewpoint_id_to_idx[viewpoint_id] = len(self.viewpoint_ids)
        self.viewpoint_ids.append(viewpoint_id)

        if not init_viewpoint:
            if self.txs_world_viewpoint:
                # use the previously added viewpoint as a guess
                init_viewpoint = self.txs_world_viewpoint[-1]
            else:
                init_viewpoint = self.init_viewpoint
        
        self.txs_world_viewpoint.append(init_viewpoint)
        self.viewpoint_infos.append(InfoState6())
        viewpoint_idx = self.viewpoint_id_to_idx[viewpoint_id]
        viewpoint_detections_start = len(self.detections)
        for tag_id, tag_corners in tags.items():
            if tag_id not in self.tag_id_to_idx:
                self.tag_id_to_idx[tag_id] = len(self.tag_ids)
                self.tag_ids.append(tag_id)
                self.txs_world_tag.append(np.eye(4))
                self.tag_infos.append(InfoState6())
            tag_idx = self.tag_id_to_idx[tag_id]
            # print("viewpoint", viewpoint_id, "contained tag at idx", tag_idx)
            dim_detection_factor_input = MapBuilder3d.DIM_VIEWPOINT_POSE + MapBuilder3d.DIM_TAG_POSE
            self.detections.append((tag_idx, viewpoint_idx, np.reshape(tag_corners, (8,1))))
            self.detection_jacobians.append(np.zeros(shape=(8,dim_detection_factor_input)))
            self.detection_projections.append(np.zeros(shape=(8,1)))
            self.detection_residuals.append(np.zeros(shape=(8,1)))
            self.detection_JtJs.append(np.zeros(shape=(dim_detection_factor_input, dim_detection_factor_input)))
            self.detection_rtJs.append(np.zeros(shape=(1,dim_detection_factor_input)))
            self.detection_to_viewpoint_msgs.append(InfoState6())
            self.detection_to_tag_msgs.append(InfoState6())
            self.detection_errors.append(float('inf'))
        viewpoint_detections_end = len(self.detections)
        self.viewpoint_detections.append((viewpoint_detections_start, viewpoint_detections_end))

    def relinearize(self):
        prev_error = self.get_total_detection_error()
        
        for i, detection in enumerate(self.detections):
            tag_idx, viewpoint_idx, tag_corners = detection
            tx_world_viewpoint = self.txs_world_viewpoint[viewpoint_idx]
            tx_world_tag = self.txs_world_tag[tag_idx]
            tx_viewpoint_tag = SE3_inv(tx_world_viewpoint) @ tx_world_tag
            image_corners, dimage_corners_dcamera, dimage_corners_dtag = project(self.camera_matrix, tx_viewpoint_tag, self.corners_mat)
            self.detection_jacobians[i][:,:6] = dimage_corners_dcamera
            self.detection_jacobians[i][:,6:] = dimage_corners_dtag

            self.detection_projections[i] = image_corners
            residual = image_corners - tag_corners
            self.detection_residuals[i] = residual

            # J = self.detection_jacobians[i]
            # self.detection_JtJs[i] = self.inverse_pixel_cov * J.T @ J
            # self.detection_rtJs[i] = self.inverse_pixel_cov * residual.T @ J

            # self.detection_errors[i] = self.inverse_pixel_cov * np.dot(residual.T, residual)[0,0]

            # caculate huber loss
            huber_k = 30
            huber_w = make_huber_mat(huber_k, residual)
            # print("huber_w\n",huber_w)

            J = self.detection_jacobians[i]
            self.detection_JtJs[i] = self.inverse_pixel_cov * J.T @ huber_w @ J
            self.detection_rtJs[i] = self.inverse_pixel_cov * residual.T @ huber_w @ J

            self.detection_errors[i] = self.inverse_pixel_cov * huber_error(huber_k, residual)
            

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

        # if curr_error < prev_error:
        #     self.regularizer *= 0.5
        # else:
        #     self.regularizer *= 3.0
            

        self.regularizer = max(self.regularizer, 10)
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
            info.matrix = self.regularizer * np.eye(MapBuilder3d.DIM_TAG_POSE)

        return curr_error < prev_error

    def get_total_detection_error(self):
        return sum(self.detection_errors)

    def update(self):
        # copy linearization point
        txs_world_viewpoint_backup = [tx.copy() for tx in self.txs_world_viewpoint]
        txs_world_tag_backup = [tx.copy() for tx in self.txs_world_tag]

        for viewpoint_idx, viewpoint_info in enumerate(self.viewpoint_infos):
            delta = np.linalg.solve(viewpoint_info.matrix, viewpoint_info.vector)
            self.txs_world_viewpoint[viewpoint_idx] = self.txs_world_viewpoint[viewpoint_idx] @ se3_exp(delta)
            # self.txs_world_viewpoint[viewpoint_idx] = heuristic_flip_tx_world_cam(self.txs_world_viewpoint[viewpoint_idx])
            fix_SE3(self.txs_world_viewpoint[viewpoint_idx])

        for tag_idx, tag_info in enumerate(self.tag_infos):
            delta = np.linalg.solve(tag_info.matrix, tag_info.vector)
            self.txs_world_tag[tag_idx] = self.txs_world_tag[tag_idx] @ se3_exp(delta)
            fix_SE3(self.txs_world_tag[tag_idx])

        tx_tag0_world = SE3_inv(self.txs_world_tag[0])

        # recenter the map around tag0
        for i, tx_world_viewpoint in enumerate(self.txs_world_viewpoint):
            self.txs_world_viewpoint[i] = tx_tag0_world @ tx_world_viewpoint
            fix_SE3(self.txs_world_viewpoint[i])

        for i, tx_world_tag in enumerate(self.txs_world_tag):
            self.txs_world_tag[i] = tx_tag0_world @ self.txs_world_tag[i]
            fix_SE3(self.txs_world_tag[i])

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

        msg = InfoState6(vector_msg, matrix_msg)
        self.tag_infos[tag_idx] -= self.detection_to_tag_msgs[detection_idx] # undo the previous message from this det
        self.detection_to_tag_msgs[detection_idx] = msg
        self.tag_infos[tag_idx] += msg # add on the current message from this det
