# WIP
from hack_sys_path import *

from gl_util import GlRgbTexture
from imgui_sdl_wrapper import ImguiSdlWrapper
from overlayable import *
from pytagmapper.data import *
from pytagmapper.geometry import *
from pytagmapper.project import project, get_corners_mat
from pytagmapper.map_builder import *
from misc import *
import math
import numpy as np
from numpy.random import default_rng
import cv2
import argparse

def overlay_polyline(overlayable, polyline, colors, thickness):
    for i in range(polyline.shape[1]):
        ni = (i + 1) % polyline.shape[1]
        overlay_line(overlayable,
                     polyline[0,i],
                     polyline[1,i],
                     polyline[0,ni],
                     polyline[1,ni],
                     colors[i],
                     thickness)

INIT_TXS_CAMERA_TAG = [
    np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 1],
        [0,  0,  0, 1]
    ]),
    np.array([
        [1,   0,  0, 0],
        [0,   0, -1, 1],
        [0,   1,  0, 1],
        [0,   0,  0, 1]
    ]),
    np.array([
        [-1,   0,  0, 0],
        [ 0,   0, -1, 1],
        [ 0,  -1,  0, 1],
        [ 0,   0,  0, 1]
    ]),    
    np.array([
        [ 0, -1,  0, 0],
        [ 0,  0, -1, 1],
        [ 1,  0,  0, 1],
        [ 0,  0,  0, 1]
    ]),
    np.array([
        [ 0,  1,  0, 0],
        [ 0,  0, -1, 1],
        [-1,  0,  0, 1],
        [ 0,  0,  0, 1]
    ]),
]

def overlay_tag(overlayable, tag_corners, tag_id = None, thickness = 1):
    side_colors = [
        imgui.get_color_u32_rgba(1,0,0,1),
        imgui.get_color_u32_rgba(0,1,0,1),
        imgui.get_color_u32_rgba(0,0,1,1),
        imgui.get_color_u32_rgba(1,1,0,1)
    ]
    
    acenter = (np.sum(tag_corners, axis=1)/4).flatten()
    for i in range(4):
        color = side_colors[i]
        p0 = tag_corners[:,i].flatten()
        p1 = tag_corners[:,(i+1)%4].flatten()
        overlay_line(overlayable,
                     p0[0], p0[1], p1[0], p1[1],
                     color, thickness)

    if tag_id is not None:
        overlay_text(overlayable, acenter[0], acenter[1], imgui.get_color_u32_rgba(1,0,0,1), str(tag_id))

def main():
    print("OK")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive map builder")
    parser.add_argument('data_dir', type=str, help='input data directory')
    args = parser.parse_args()

    rng = default_rng()
    source_dir = args.data_dir
    
    images = load_images(source_dir)
    image_ids = list(reversed(sorted(images.keys())))
    data = load_data(source_dir)

    map_builder = MapBuilder(data["camera_matrix"],
                             data["tag_side_lengths"],
                             '3d')

    added_image_ids = []
    next_image_idx = 0

    app = ImguiSdlWrapper('Interactive Optimizer', 1280, 720)

    side_colors = [
        imgui.get_color_u32_rgba(1,0,0,1),
        imgui.get_color_u32_rgba(0,1,0,1),
        imgui.get_color_u32_rgba(0,0,1,1),
        imgui.get_color_u32_rgba(1,1,0,1)
    ]

    for image_id, image in images.items():
        app.add_image(image_id, image)

    optimize = False
    show_detects = True
    show_projections = True
    map_builder_step = 1

    wx = 0
    wy = 0
    wz = 0
    perturb_tag_idx = None

    update_viewpoint_idxs = set()
    update_tag_idxs = set()

    essential_matrices = {}
    essential_rs = {}
    essential_ts = {}
    essential_ps = {}

    while app.running:
        app.main_loop_begin()
        imgui.begin("control")
        imgui.text("camera matrix")
        imgui.text(str(data["camera_matrix"]))
        
        _, optimize = imgui.checkbox("optimize", optimize)
        _, show_detects = imgui.checkbox("show detects", show_detects)
        _, show_projections = imgui.checkbox("show projections", show_projections)

        imgui.text(f"error {map_builder.get_total_detection_error()}")
        imgui.text(f"reg {map_builder.regularizer}")

        if imgui.button("decimate regularizer"):
            map_builder.regularizer *= 0.1

        if len(added_image_ids) != len(image_ids) and \
           imgui.button("add image"):
            image_id = image_ids[next_image_idx]
            added_image_ids.append(image_id)
            map_builder.add_viewpoint(image_id, data["viewpoints"][image_id])
            map_builder.relinearize()
            next_image_idx += 1

        if optimize:
            map_builder.send_detection_to_viewpoint_msgs()
            map_builder.send_detection_to_tag_msgs()
            if map_builder_step % 30 == 0:
                map_builder.update()
            map_builder_step += 1
        imgui.end()

        imgui.begin("images")
        display_width = imgui.get_window_width() - 10
        for image_idx, image_id in enumerate(added_image_ids):
            imgui.push_id(str(image_idx))
            imgui.text(f"image {image_id}")            

            if image_idx != 0 and image_id not in essential_matrices:
                if imgui.button("compute essential matrix"):
                    # build point correspondences
                    other_image_id = added_image_ids[0]
                    other_viewpoint = data["viewpoints"][other_image_id]
                    this_viewpoint = data["viewpoints"][image_id]
                    same_tag_ids = other_viewpoint.keys() & this_viewpoint.keys()

                    inv_camera_matrix = np.linalg.inv(map_builder.camera_matrix)

                    pts1 = []
                    pts2 = []

                    for tag_id in same_tag_ids:
                        for i in range(4):
                            this_xy = this_viewpoint[tag_id][2*i:2*i+2]
                            other_xy = other_viewpoint[tag_id][2*i:2*i+2]
                            pts1.append(other_xy)
                            pts2.append(this_xy)

                    pts1 = np.array(pts1)
                    pts2 = np.array(pts2)

                    pts1_normalized = np.empty((3, 12))
                    pts1_normalized[2,:] = 1
                    pts1_normalized[:2,:] = pts1.T
                    pts1_normalized = inv_camera_matrix @ pts1_normalized

                    pts2_normalized = np.empty((3, 12))
                    pts2_normalized[2,:] = 1
                    pts2_normalized[:2,:] = pts2.T
                    pts2_normalized = inv_camera_matrix @ pts2_normalized

                    essential_ps[image_id] = (pts1_normalized, pts2_normalized)

                    e, mask = cv2.findEssentialMat(pts1, pts2, map_builder.camera_matrix)

                    for i in range(pts1.shape[0]):
                        print("checking essential",i)
                        pt1 = pts1_normalized[:,i]
                        pt2 = pts2_normalized[:,i]
                        print("pt1", pt1.T)
                        print("pt2", pt2.T)
                        res = pt2.T @ e @ pt1
                        print("\tres", res)
                    
                    essential_matrices[image_id] = e
                    result = cv2.recoverPose(e, pts1, pts2, map_builder.camera_matrix, distanceThresh=100)
                    R = result[1]
                    t = result[2]
                    essential_rs[image_id] = R
                    essential_ts[image_id] = t
                    mask = result[3]
                    triangulations = result[4]
                    triangulations /= triangulations[3,:]

                    tx_othercam_thiscam = np.zeros((4,4))
                    tx_othercam_thiscam[:3,:3] = R
                    tx_othercam_thiscam[:3,3] = t.T
                    tx_othercam_thiscam[3,3] = 1

                    # project triangulations into othercam
                    projected = triangulations[:3,:].copy()
                    projected /= projected[2,:]

                    projected2 = SE3_inv(tx_othercam_thiscam) @ triangulations
                    projected2 = projected2[:3,:].copy()
                    projected2 /= projected2[2,:]

                    print("projected\n", projected.T)
                    print("pts1 normalized\n", pts1_normalized.T)
                    
                    print("projected2\n", projected2.T)
                    print("pts2 normalized\n", pts2_normalized.T)
                    
                    print("tx_othercam_thiscam\n", tx_othercam_thiscam)
                    print("recoverpose result\n", result)
                    print("triangulations shape\n", triangulations.shape)
                    print("triangulations\n", triangulations.T)

            if image_id in essential_matrices:
                imgui.text("e")
                imgui.text(str(essential_matrices[image_id]))
                # compute tx_other_this

                R = essential_rs[image_id]
                t = essential_ts[image_id]
                tx_othercam_thiscam = np.zeros((4,4))
                tx_othercam_thiscam[:3,:3] = R
                tx_othercam_thiscam[:3,3] = t.T
                tx_othercam_thiscam[3,3] = 1

                imgui.text("tx_othercam_thiscam from essential")
                imgui.text(str(tx_othercam_thiscam))

                # get othercam thiscam from map_builder
                tx_world_othercam = map_builder.txs_world_viewpoint[0]
                tx_world_thiscam = map_builder.txs_world_viewpoint[map_builder.viewpoint_id_to_idx[image_id]]

                tx_othercam_thiscam_mb = SE3_inv(tx_world_othercam) @ tx_world_thiscam

                R_mb = tx_othercam_thiscam_mb[:3,:3].copy()
                t_mb = tx_othercam_thiscam[:3,3].copy()
                tx_mb = so3_to_matrix(np.array([t_mb]).T)
                e_mb = tx_mb @ R_mb

                imgui.text("essential e_mb")
                imgui.text(str(e_mb))

                p1s, p2s = essential_ps[image_id]
                for i in range(p1s.shape[1]):
                    pt1 = p1s[:,i]
                    pt2 = p2s[:,i]
                    res = pt1.T @ e_mb @ pt2
                    imgui.text(f"\tres{i}: p1 {pt1.T} p2 {pt2.T} res {res}")

                imgui.text("tx_othercam_thiscam from mb")
                imgui.text(str(tx_othercam_thiscam_mb))
                imgui.text("tx_thiscam_othercam from mb")                
                imgui.text(str(SE3_inv(tx_othercam_thiscam_mb)))
                

            for i, tx_camera_world in enumerate(INIT_TXS_CAMERA_TAG):
                if imgui.button(f"reinit {i}"):
                    map_builder.txs_world_viewpoint[image_idx] = tx_camera_world
                if i+1 != len(INIT_TXS_CAMERA_TAG):
                    imgui.same_line()
            
            should_update = image_idx in update_viewpoint_idxs
            _, should_update = imgui.checkbox("update viewpoint", should_update)
            if should_update:
                update_viewpoint_idxs.add(image_idx)
            elif image_idx in update_viewpoint_idxs:
                update_viewpoint_idxs.remove(image_idx)

            if image_idx in update_viewpoint_idxs:
                map_builder.update_viewpoint(image_idx)

            tx_viewpoint_world = SE3_inv(map_builder.txs_world_viewpoint[image_idx])
            
            for det_idx in range(*map_builder.viewpoint_detections[image_idx]):
                detection = map_builder.detections[det_idx]
                tag_idx, viewpoint_idx, tag_corners = detection
                tag_id = map_builder.tag_ids[tag_idx]

                update_tag = tag_idx in update_tag_idxs
                _, update_tag = imgui.checkbox(f"update tag {tag_id}", update_tag)
                imgui.same_line()
                if imgui.button(f"reinit tag {tag_id}"):
                    map_builder.txs_world_tag[tag_idx] = np.eye(map_builder.tx_world_tag_dim)

                if update_tag:
                    update_tag_idxs.add(tag_idx)
                elif tag_idx in update_tag_idxs:
                    update_tag_idxs.remove(tag_idx)

                if tag_idx in update_tag_idxs:
                    map_builder.update_tag(tag_idx)

            if imgui.button("update none"):
                update_tag_idxs = set()
                update_viewpoint_idxs = set()

            imgui.text("tx_world_viewpoint")
            imgui.text(str(map_builder.txs_world_viewpoint[image_idx]))

            tid, w, h = app.get_image(image_id)
            image = draw_overlayable_image(tid, w, h, display_width)
            sx, sy = 0, 0
            clicked = False
            tag_clicked = None

            if imgui.is_item_hovered():
                mx, my = imgui.get_mouse_pos()
                sx, sy = overlay_inv_transform(image, mx, my)
                clicked = imgui.is_mouse_clicked()

            for det_idx in range(*map_builder.viewpoint_detections[image_idx]):
                detection = map_builder.detections[det_idx]
                tag_idx, viewpoint_idx, tag_corners = detection
                tag_id = map_builder.tag_ids[tag_idx]

                projections = map_builder.detection_projections[det_idx]

                if show_projections:
                    overlay_tag(image, projections.reshape((2,4), order='F'), tag_id)

                if show_detects:
                    overlay_tag(image, tag_corners.reshape((2,4), order='F'), tag_id)

                if quad_contains_pt(projections.reshape((2,4), order='F'), (sx, sy)):
                    # calculate tx_camera_tag
                    tx_viewpoint_tag = tx_viewpoint_world @ map_builder.txs_world_tag[tag_idx]
                    
                    imgui.set_tooltip(f"{tag_id}\n{tx_viewpoint_tag}")
                    if imgui.is_mouse_clicked():
                        tag_clicked = tag_idx

                if tag_idx == perturb_tag_idx:
                    delta = np.zeros((6,1))
                    delta[0,0] = wx
                    delta[1,0] = wy
                    delta[2,0] = wz
                    ptx_viewpoint_tag = tx_viewpoint_world @ map_builder.txs_world_tag[tag_idx] @ se3_exp(delta)
                    corners = map_builder.camera_matrix @ (ptx_viewpoint_tag @ map_builder.corners_mats[tag_idx])[:3,:]
                    corners[:2,:] /= corners[2,:]
                    overlay_polyline(image, corners[:2,:], side_colors, 1)
                    

            if clicked:
                if perturb_tag_idx != tag_clicked:
                    wx = 0
                    wy = 0
                    wz = 0
                    perturb_tag_idx = tag_clicked

            imgui.pop_id()

        if not perturb_tag_idx is None:
            if not imgui.begin("tag perturb"):
                perturb_tag_idx = None
                wx = 0
                wy = 0
                wz = 0
                imgui.end()
            else:
                imgui.text(str(map_builder.tag_ids[perturb_tag_idx]))
                
                _, wx = imgui.slider_float("wx", wx, -math.pi, math.pi)
                _, wy = imgui.slider_float("wy", wy, -math.pi, math.pi)
                _, wz = imgui.slider_float("wz", wz, -math.pi, math.pi)
                if imgui.button("clear"):
                    wx = 0
                    wy = 0
                    wz = 0
                if imgui.button("apply"):
                    delta = np.zeros((6,1))
                    delta[0,0] = wx
                    delta[1,0] = wy
                    delta[2,0] = wz
                    map_builder.txs_world_tag[perturb_tag_idx] =  map_builder.txs_world_tag[perturb_tag_idx] @ se3_exp(delta)
                    wx = 0
                    wy = 0
                    wz = 0
                imgui.end()

        imgui.end()
        app.main_loop_end()
    app.destroy()

    
