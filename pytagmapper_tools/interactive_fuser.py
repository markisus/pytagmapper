from hack_sys_path import *

import copy
from gl_util import GlRgbTexture
from imgui_sdl_wrapper import ImguiSdlWrapper
from misc import *
from numpy.random import default_rng
from overlayable import *
from pytagmapper.data import *
from pytagmapper.geometry import *
from pytagmapper.inside_out_tracker import InsideOutTracker
from pytagmapper.map_builder import *
from pytagmapper.project import project, get_corners_mat
import argparse
import cv2
import math
import numpy as np
from collections import deque, defaultdict

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

def get_corners_mat_from_map(map_data, tag_id):
    default_tag_side_length = map_data["tag_side_lengths"]["default"]
    tag_side_length = map_data["tag_side_lengths"].get(tag_id, default_tag_side_length)
    return get_corners_mat(tag_side_length)

def init_map_builder(camera_matrix, tag_side_lengths, map_ids, maps, viewpoints, source_data, source_tag_ids, txs_world_map, txs_world_viewpoint):
    map_builder = MapBuilder(camera_matrix, tag_side_lengths, map_type='3d')
    map_builder.huber_k = float('inf') # no outliers
    txs_world_tag = {}
    for map_id in map_ids:
        tx_world_map = txs_world_map[map_id]
        for tag_id, tx_map_tag in maps[map_id]["tag_locations"].items():
            txs_world_tag[tag_id] = tx_world_map @ tx_map_tag

    for image_id in image_ids:
        tags = data["viewpoints"][image_id]
        # use only those tags which appear in the source maps
        tags = {k: v for k, v in tags.items() if k in source_tag_ids} 
        map_builder.add_viewpoint(image_id, tags,
                                  init_viewpoint = txs_world_viewpoint[image_id],
                                  init_tags = txs_world_tag)

    # add the viewpoints from the source data sets
    for map_id in map_ids:
        for image_id, tags in source_data[map_id]["viewpoints"].items():
            tx_map_viewpoint = viewpoints[map_id][image_id]
            tx_world_viewpoint = txs_world_map[map_id] @ tx_map_viewpoint
            map_builder.add_viewpoint(f"{map_id}_{image_id}", tags,
                                      init_viewpoint = tx_world_viewpoint,
                                      init_tags = txs_world_tag)

    map_builder.relinearize()
    return map_builder
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fusion_dir", type=str, help="directory containing data used to fuse the other maps together")
    parser.add_argument("map_dirs", type=str, nargs="+", help="directory containing maps to be fused")
    parser.add_argument("--allow-new-tags", "-a", action="store_true", default=False)
    args = parser.parse_args()

    map_ids = args.map_dirs
    maps = { map_id : load_map(map_id) for map_id in map_ids }
    camera_matrix = load_camera_matrix(map_ids[0]) # assume all camera matrices are the same

    data = load_data(args.fusion_dir)
    images = load_images(args.fusion_dir)
    image_ids = sorted(images.keys())    

    map_ids = sorted(maps.keys())
    source_viewpoints = { map_id : load_viewpoints(map_id) for map_id in map_ids }
    source_data = { map_id : load_data(map_id) for map_id in map_ids }
    source_images = { map_id : load_images(map_id) for map_id in map_ids }

    tag_side_lengths = {}
    for map_id, map in maps.items():
        tag_side_lengths.update(copy.deepcopy(map["tag_side_lengths"]))

    map_builder = MapBuilder(camera_matrix, tag_side_lengths, map_type='3d')

    source_tag_ids = set()
    for map in maps.values():
        source_tag_ids.update(map["tag_locations"].keys())

    if args.allow_new_tags:
        for viewpoints in data["viewpoints"].values():
            source_tag_ids.update(viewpoints.keys())

    app = ImguiSdlWrapper('Interactive Fuser', 1280, 720)

    side_colors = [
        imgui.get_color_u32_rgba(1,0,0,1),
        imgui.get_color_u32_rgba(0,1,0,1),
        imgui.get_color_u32_rgba(0,0,1,1),
        imgui.get_color_u32_rgba(1,1,0,1)
    ]

    for image_id, image in images.items():
        app.add_image(image_id, image)
    for map_id, images in source_images.items():
        for image_id, image in images.items():
            app.add_image(f"{map_id}_{image_id}", image)

    show_mb = False
    show_fused = True
    show_iotrackers = False
    optimize = False
    optimize_mb = False
    step = 0

    # phases
    FUSION_PHASE = "fusion_phase"
    REFINEMENT_PHASE = "refinement_phase"
    phase = FUSION_PHASE

    image_map_overlaps = defaultdict(list)
    for image_id in image_ids:
        for map_id in map_ids:
            overlap = maps[map_id]["tag_locations"].keys() & data["viewpoints"][image_id].keys()
            if len(overlap) != 0:
                image_map_overlaps[image_id].append(map_id)

    # build plan does breadth first search to add viewpoints into the map builder
    # first pass, add in an arbitrary map (map0)
    # add all viewpoints that contain this map, and queue next maps

    image_plan = []
    map_queue = deque((map_ids[0],))
    while map_queue:
        map_id = map_queue.popleft()
        # print("Handling images overlapping map", map_id)
        # add all images which overlap this map
        for image_id in image_ids:
            if image_id in image_plan:
                # this image is already in the build plan
                continue
            # print("\timage", image_id, "overlaps this map")
            if map_id in image_map_overlaps[image_id]:
                image_plan.append(image_id)
                # queue the maps of this image
                for other_map_id in image_map_overlaps[image_id]:
                    if other_map_id not in map_queue and other_map_id != map_id:
                        # print("\timage", image_id, "causing map", other_map_id, "to be queued")
                        map_queue.append(other_map_id)

    iotrackers = {}
    # for (viewpoint_id, map_id) => init an inside out tracker

    for map_id, map in maps.items():
        for image_id in image_ids:
            overlap = map["tag_locations"].keys() & data["viewpoints"][image_id].keys()
            if len(overlap) == 0:
                # image does not contain any detections from this map
                continue
            
            iotrackers[(image_id, map_id)] = InsideOutTracker(
                camera_matrix, map)

    while app.running:
        if optimize and phase == REFINEMENT_PHASE:
            if step % 20 == 0:
                map_builder.update()
            else:
                map_builder.send_detection_to_viewpoint_msgs()
                map_builder.send_detection_to_tag_msgs()
            step += 1

        # FUSION STATE
        txs_world_map = { map_ids[0]: np.eye(4) }
        txs_world_viewpoint = { }
        for image_id in image_plan:
            # this viewpoint has some link to the world due to the txs_world_map
            for map_id in image_map_overlaps[image_id]:
                if map_id in txs_world_map:
                    iotracker = iotrackers[(image_id, map_id)]
                    tx_map_viewpoint = iotracker.tx_world_viewpoint
                    tx_world_map = txs_world_map[map_id]
                    tx_world_viewpoint = tx_world_map @ tx_map_viewpoint
                    txs_world_viewpoint[image_id] = tx_world_viewpoint
                    for other_map_id in image_map_overlaps[image_id]:
                        # the other maps visible from this view imply
                        # the positions of the other maps in the world
                        if other_map_id not in txs_world_map:
                            iotracker = iotrackers[(image_id, other_map_id)]
                            tx_othermap_viewpoint = iotracker.tx_world_viewpoint
                            tx_viewpoint_othermap = SE3_inv(tx_othermap_viewpoint)
                            tx_world_othermap = tx_world_viewpoint @ tx_viewpoint_othermap
                            txs_world_map[other_map_id] = tx_world_othermap

        app.main_loop_begin()
        imgui.begin("Control")
        imgui.text(f"phase {phase}")        
        _, optimize = imgui.checkbox("optimize", optimize)
        if phase == FUSION_PHASE:
            _, show_fused = imgui.checkbox("show fused", show_fused)
            _, show_iotrackers = imgui.checkbox("show iotrackers", show_iotrackers)
            if imgui.button("mark all converged"):
                for iotracker in iotrackers.values():
                    iotracker.converged_guess = iotracker.best_guess

        if phase == REFINEMENT_PHASE:
            show_fused = False
            show_iotrackers = False
            


        show_mb = phase == REFINEMENT_PHASE

        if phase == REFINEMENT_PHASE:
            imgui.text(f"map builder error {map_builder.get_total_detection_error():#.4g}")
        if phase == FUSION_PHASE:
            error = 0
            for iotracker in iotrackers.values():
                error += iotracker.error
            imgui.text(f"total tracker error {error:#.4g}")

        if phase == REFINEMENT_PHASE and imgui.button("save map"):
            save_map3d_json(
                args.fusion_dir,
                map_builder.tag_side_lengths,
                map_builder.tag_ids,
                map_builder.txs_world_tag)
            save_viewpoints_json(
                args.fusion_dir,
                map_builder.viewpoint_ids,
                map_builder.txs_world_viewpoint)
            

        imgui.end()

        imgui.begin("Source Maps")
        display_width = imgui.get_window_width() - 10
        for map_id in map_ids:
            for image_id in sorted(source_images[map_id].keys()):
                image_id = f"{map_id}_{image_id}" # extend image id with source map
                imgui.text(image_id)
                tid, w, h = app.get_image(image_id)
                image = draw_overlayable_image(tid, w, h, display_width)

                # project map builder detections into this image
                if show_mb and image_id in map_builder.viewpoint_id_to_idx:
                    viewpoint_idx = map_builder.viewpoint_id_to_idx[image_id]
                    tx_world_viewpoint = map_builder.txs_world_viewpoint[viewpoint_idx]
                    tx_viewpoint_world = SE3_inv(tx_world_viewpoint)

                    for tag_id, tag_idx in map_builder.tag_id_to_idx.items():
                        tx_world_tag = map_builder.txs_world_tag[tag_idx]
                        tx_viewpoint_tag = tx_viewpoint_world @ tx_world_tag

                        tag_dir = tx_viewpoint_tag[:3,2].copy()
                        to_tag = tx_viewpoint_tag[:3,3].copy()
                        to_tag /= np.linalg.norm(to_tag)
                        tag_dp = np.dot(tag_dir, to_tag)
                        if tag_dp >= 0:
                            # camera is looking at the back of tag
                            continue
                        
                        corners_mat = map_builder.corners_mats[tag_idx]
                        projection = camera_matrix @ (tx_viewpoint_tag @ corners_mat)[:3,:]
                        projection /= projection[2,:]
                        overlay_polyline(image, projection[:2,:], side_colors, 1)
                        center = np.mean(projection, axis=1)
                        overlay_text(image, center[0], center[1], side_colors[0], str(tag_id))
                
        imgui.end()

        imgui.begin("Links")

        fusion_converged = True
        display_width = imgui.get_window_width() - 10
        for image_id in image_ids:
            tid, w, h = app.get_image(image_id)
            image = draw_overlayable_image(tid, w, h, display_width)
            tags = data["viewpoints"][image_id]

            # project map builder detections into this image
            if show_mb and image_id in map_builder.viewpoint_id_to_idx:
                viewpoint_idx = map_builder.viewpoint_id_to_idx[image_id]
                tx_world_viewpoint = map_builder.txs_world_viewpoint[viewpoint_idx]
                tx_viewpoint_world = SE3_inv(tx_world_viewpoint)

                for tag_id, tag_idx in map_builder.tag_id_to_idx.items():
                    tx_world_tag = map_builder.txs_world_tag[tag_idx]
                    tx_viewpoint_tag = tx_viewpoint_world @ tx_world_tag

                    tag_dir = tx_viewpoint_tag[:3,2].copy()
                    to_tag = tx_viewpoint_tag[:3,3].copy()
                    to_tag /= np.linalg.norm(to_tag)
                    tag_dp = np.dot(tag_dir, to_tag)
                    if tag_dp >= 0:
                        # camera is looking at the back of tag
                        continue

                    corners_mat = map_builder.corners_mats[tag_idx]
                    projection = camera_matrix @ (tx_viewpoint_tag @ corners_mat)[:3,:]
                    projection /= projection[2,:]
                    overlay_polyline(image, projection[:2,:], side_colors, 1)
                    center = np.mean(projection, axis=1)
                    overlay_text(image, center[0], center[1], side_colors[0], str(tag_id))

            # project maps into this image
            if show_fused:
                for map_id in map_ids:
                    if map_id not in image_map_overlaps[image_id]:
                        continue
                    tx_world_viewpoint = txs_world_viewpoint[image_id]
                    tx_world_map = txs_world_map[map_id]
                    tx_viewpoint_map = SE3_inv(tx_world_viewpoint) @ tx_world_map
                    for tag_id, tx_map_tag in maps[map_id]["tag_locations"].items():
                        corners_mat = get_corners_mat_from_map(maps[map_id], tag_id)
                        tx_viewpoint_tag = tx_viewpoint_map @ tx_map_tag
                        projection = camera_matrix @ (tx_viewpoint_tag @ corners_mat)[:3,:]
                        projection /= projection[2,:]
                        overlay_polyline(image, projection[:2,:], side_colors, 1)
                        center = np.mean(projection, axis=1)
                        overlay_text(image, center[0], center[1], side_colors[0], str(tag_id))

            # using each inside out tracker, project tags into image
            for (image_id0, map_id), iotracker in iotrackers.items():
                if image_id0 != image_id:
                    continue

                imgui.push_id(f"{image_id0}_{map_id}")                
                if iotracker.converged_guess is None:
                    if imgui.button("mark converged"):
                        iotracker.converged_guess = iotracker.best_guess
                    else:
                        fusion_converged = False
                    imgui.same_line()

                imgui.text(f"{map_id} error {iotracker.error:#.4g}, converged {iotracker.converged_guess}")

                if optimize and phase == FUSION_PHASE:
                    iotracker.update1(tags.items(), force_update=False)

                if show_iotrackers:
                    proj_ids, projections = iotracker.get_projections()
                    for tag_id, proj in zip(proj_ids, projections):
                        proj = proj.reshape((2, 4), order='F')
                        overlay_polyline(image, proj, side_colors, 1)
                        center = np.mean(proj, axis=1)
                        overlay_text(image, center[0], center[1], side_colors[0], str(tag_id))

                imgui.pop_id()

        if fusion_converged and phase == FUSION_PHASE:
            phase = REFINEMENT_PHASE
            map_builder = init_map_builder(camera_matrix, tag_side_lengths, map_ids, maps, source_viewpoints,
                                           source_data, source_tag_ids,
                                           txs_world_map, txs_world_viewpoint)

        imgui.end()
        app.main_loop_end()
    app.destroy()

    
