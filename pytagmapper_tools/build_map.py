from hack_sys_path import *
import argparse
import random
from pytagmapper import data
from pytagmapper import project
from pytagmapper.geometry import *
from pytagmapper.map_builder import MapBuilder
import sys

import cv2
import numpy as np

def solvePnPWrapper(obj_points, img_points, camera_matrix):
    succ, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, None)
    if not succ:
        raise RuntimeError("solvePnP failed")
    rot, _ = cv2.Rodrigues(rvec)
    tx_camera_obj = np.eye(4, dtype=np.float64)
    tx_camera_obj[:3,:3] = rot
    tx_camera_obj[:3,3:4] = tvec
    return tx_camera_obj

def add_viewpoint(source_data, viewpoint_id, map_builder, total_viewpoints):
    viewpoint = source_data['viewpoints'][viewpoint_id]

    map_builder.add_viewpoint(viewpoint_id,
                              viewpoint)

    map_builder.relinearize()

    error = map_builder.get_avg_detection_error()
    change_pct = 1
    improved = False
    num_its = 0

    try:
        while (improved and change_pct >= 1e-3) or error >= 0.5 or num_its <= 3:
            print(f"[{len(map_builder.viewpoint_ids)}/{total_viewpoints}] change {change_pct*100:#.4g}% error {error:#.4g}\r", end='')            
            sys.stdout.flush()
            prev_error = error
            for i in range(20):
                map_builder.send_detection_to_viewpoint_msgs()
                map_builder.send_detection_to_tag_msgs()
            improved = map_builder.update()
            error = map_builder.get_avg_detection_error()
            delta = max(prev_error - error, 0)
            change_pct = delta/prev_error
            num_its += 1

    except KeyboardInterrupt:
        pass
    
    return map_builder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster images based on shared tags")
    parser.add_argument('directory', type=str, help='scene data directory')
    parser.add_argument('--output-dir', '-o', type=str, help='output data directory', default='')
    parser.add_argument('--mode', type=str, default='3d', help='2d, 2.5d, or 3d (default 3d)')
    args = parser.parse_args()

    if args.mode not in ['2.5d', '3d', '2d']:
        raise RuntimeError("Unexpected map type", args.mode)

    output_dir = args.output_dir or args.directory
    scene_data = data.load_data(args.directory)

    viewpoints = scene_data['viewpoints']
    viewpoint_ids = list(viewpoints.keys())
    random.shuffle(viewpoint_ids)
    # print(f"viewpoint ids {viewpoint_ids}")

    used_viewpoints = set()

    # get the image with the most tags
    best_viewpoint = 0
    best_num_tags = -1
    for viewpoint_id in viewpoint_ids:
        if len(viewpoints[viewpoint_id]) > best_num_tags:
            best_num_tags = len(viewpoints[viewpoint_id])
            best_viewpoint = viewpoint_id

    # print("best viewpoint was ", best_viewpoint)
    # print("best num tags ", best_num_tags)

    map_builder = MapBuilder(scene_data['camera_matrix'],
                             scene_data['tag_side_lengths'],
                             args.mode)

    map_builder.add_viewpoint(best_viewpoint,
                              scene_data['viewpoints'][best_viewpoint])
    map_builder.relinearize()
    used_viewpoints.add(best_viewpoint)

    print("Optimizing viewpoint. ctrl+c to skip.")

    while viewpoints.keys() - used_viewpoints:
        # find the viewpoint with the most overlap with the map
        best_viewpoint = -1
        best_overlap = {}
        for viewpoint_id in viewpoint_ids:
            if viewpoint_id in used_viewpoints:
                continue
            overlap = viewpoints[viewpoint_id].keys() & map_builder.tag_id_to_idx.keys()
            if len(overlap) > len(best_overlap):
                best_overlap = overlap
                best_viewpoint = viewpoint_id

        # print("best overlap from viewpoint", best_viewpoint, "of len", len(best_overlap))
        add_viewpoint(scene_data, best_viewpoint, map_builder, len(scene_data['viewpoints']))
        used_viewpoints.add(best_viewpoint)

    error = map_builder.get_avg_detection_error()
    change_pct = 1
    improved = False
    num_its = 0
    try:
        while (improved and change_pct >= 1e-4) or error >= 0.1 or num_its < 3:
            print(f"[final] change {change_pct*100:#.4g}% error {error:#.4g}\r", end='')
            sys.stdout.flush()
            prev_error = error
            for i in range(20):
                map_builder.send_detection_to_viewpoint_msgs()
                map_builder.send_detection_to_tag_msgs()
            improved = map_builder.update()
            error = map_builder.get_avg_detection_error()
            delta = max(prev_error - error, 0)
            change_pct = delta/prev_error
            num_its += 1

            if (improved and change_pct <= 1e-5):
                # give up
                break
    except KeyboardInterrupt:
        pass

    print("\r" + " "*100, end='') # clear out the loading bar
    print("\rSaving to", output_dir)
    data.save_viewpoints_json(
        output_dir,
        map_builder.viewpoint_ids,
        map_builder.txs_world_viewpoint)

    if args.mode == '3d':
        data.save_map3d_json(
            output_dir,
            map_builder.tag_side_lengths,
            map_builder.tag_ids,
            map_builder.txs_world_tag)
    elif args.mode == '2.5d':
        data.save_map2p5d_json(
            output_dir,
            map_builder.tag_side_lengths,
            map_builder.tag_ids,
            map_builder.txs_world_tag)
    else:
        data.save_map_json(
            output_dir,
            map_builder.tag_side_lengths,
            map_builder.tag_ids,
            map_builder.txs_world_tag)    
