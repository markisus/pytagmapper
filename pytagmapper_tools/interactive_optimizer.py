# WIP
from hack_sys_path import *

import random
from gl_util import GlRgbTexture
from imgui_sdl_wrapper import ImguiSdlWrapper
from overlayable import *
from pytagmapper.data import *
from pytagmapper.geometry import *
from pytagmapper.project import project_points
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

def overlay_coordinate_frame(overlayable, corners, frame_id, thickness = 1):
    side_colors = [
        imgui.get_color_u32_rgba(1,0,0,1),
        imgui.get_color_u32_rgba(0,1,0,1),
        imgui.get_color_u32_rgba(0,0,1,1),
    ]

    for i in range(3):
        color = side_colors[i]
        p0 = corners[:,0].flatten()
        p1 = corners[:,i+1].flatten()
        overlay_line(overlayable,
                     p0[0], p0[1], p1[0], p1[1],
                     color, thickness)

    acenter = (np.sum(corners, axis=1)/4).flatten()        
    if frame_id is not None:
        overlay_text(overlayable, acenter[0], acenter[1], imgui.get_color_u32_rgba(1,0,0,1), str(frame_id))


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
    parser.add_argument('--mode', type=str, default='3d')

    args = parser.parse_args()

    rng = default_rng()
    source_dir = args.data_dir
    
    images = load_images(source_dir)
    image_ids = list(reversed(sorted(images.keys())))
    data = load_data(source_dir)

    map_builder = MapBuilder(data["camera_matrix"],
                             data["tag_side_lengths"],
                             args.mode)

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

    worldcam_x = 0
    worldcam_y = 0
    worldcam_z = 1.5
    worldcam_rx = 0
    worldcam_ry = 0
    worldcam_rz = 0

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
            # map_builder.send_detection_to_viewpoint_msgs()
            # map_builder.send_detection_to_tag_msgs()
            # if map_builder_step % 10 == 0:
            #     map_builder.update()
            map_builder.prioritized_update()
            map_builder_step += 1
        imgui.end()

        txs_world_tag = map_builder.txs_world_tag
        txs_world_viewpoint = map_builder.txs_world_viewpoint

        if imgui.begin("map"):
            display_width = imgui.get_window_width() - 10
            display = draw_overlayable_rectangle(1000, 1000, display_width)

            _, worldcam_x = imgui.slider_float("worldcam x", worldcam_x, -1, 1)
            _, worldcam_y = imgui.slider_float("worldcam y", worldcam_y, -1, 1)
            _, worldcam_z = imgui.slider_float("worldcam z", worldcam_z, 0.1, 5)
            _, worldcam_rx = imgui.slider_float("worldcam rx", worldcam_rx, -1, 1)
            _, worldcam_ry = imgui.slider_float("worldcam ry", worldcam_ry, -1, 1)
            _, worldcam_rz = imgui.slider_float("worldcam rz", worldcam_rz, -1, 1)
            
            tx_world_worldcam = np.array([
                [-1, 0, 0, worldcam_x],
                [0, 1, 0, worldcam_y],
                [0, 0, -1, worldcam_z],
                [0, 0, 0, 0],
            ])

            tx_worldrot_world = se3_exp(np.array([[worldcam_rx, worldcam_ry, worldcam_rz, 0, 0, 0]]).T)
            tx_world_worldcam = tx_worldrot_world @ tx_world_worldcam

            se3_world_worldcam = SE3_log(tx_world_worldcam)

            for tag_idx, se3_world_tag in enumerate(map_builder.se3s_world_tag):
                points, _, _ = project_points(map_builder.camparams,
                                              se3_world_worldcam,
                                              se3_world_tag,
                                              map_builder.corners_mats[tag_idx])
                # print("points for tag ", map_builder.tag_ids[tag_idx], "was ", points.T)
                # print("tx world tag was\n", tx_world_tag)
                # print("se3 world tag was\n", se3_world_tag.T)
                overlay_tag(display, points.reshape((2,4), order='F'), map_builder.tag_ids[tag_idx])


            for viewpoint_idx, se3_world_viewpoint in enumerate(map_builder.se3s_world_viewpoint):
                # xyzw format
                frame = np.array([
                    [0.0, 0.0, 0.0, 1.0],
                    [0.01, 0.0, 0.0, 1.0],
                    [0.0, 0.01, 0.0, 1.0],
                    [0.0, 0.0, 0.01, 1.0]
                ]).T
                frame_projected, _, _ = project_points(map_builder.camparams,
                                                       se3_world_worldcam,
                                                       se3_world_viewpoint,
                                                       frame)                                                       
                overlay_coordinate_frame(display, frame_projected.reshape((2,4), order='F'), map_builder.viewpoint_ids[viewpoint_idx])
                                                       
            imgui.end()

        imgui.begin("images")
        display_width = imgui.get_window_width() - 10
        for image_idx, image_id in enumerate(added_image_ids):
            imgui.push_id(str(image_idx))
            imgui.text(f"image {image_id}")            

            tx_viewpoint_world = SE3_inv(txs_world_viewpoint[image_idx])
            
            imgui.text("tx_world_viewpoint")
            imgui.text(str(txs_world_viewpoint[image_idx]))

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
                    tx_viewpoint_tag = tx_viewpoint_world @ txs_world_tag[tag_idx]
                    
                    imgui.set_tooltip(f"{tag_id}\n\ntx_world_tag\n{txs_world_tag[tag_idx]}\n\ntx_viewpoint_tag\n{tx_viewpoint_tag}")
                    if imgui.is_mouse_clicked():
                        tag_clicked = tag_idx

                if tag_idx == perturb_tag_idx:
                    delta = np.zeros((6,1))
                    delta[0,0] = wx
                    delta[1,0] = wy
                    delta[2,0] = wz
                    ptx_viewpoint_tag = tx_viewpoint_world @ txs_world_tag[tag_idx] @ se3_exp(delta)
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
                tag_id = map_builder.tag_ids[perturb_tag_idx]
                imgui.text(str(tag_id))
                
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
                    map_builder.reset_tag(tag_id, txs_world_tag[perturb_tag_idx] @ se3_exp(delta))
                    # txs_world_tag[perturb_tag_idx] =  txs_world_tag[perturb_tag_idx] @ se3_exp(delta)
                    wx = 0
                    wy = 0
                    wz = 0
                imgui.end()

        imgui.end()
        app.main_loop_end()
    app.destroy()

    
