from pytagmapper.data import *
from pytagmapper.geometry import *
from pytagmapper.map_builder import *
from pytagmapper.project import *
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Build a map from images of tags.')
    parser.add_argument('data_dir', type=str, help='input data directory')
    parser.add_argument('--output-dir', '-o', type=str, help='output data directory', default='')
    parser.add_argument('--mode', type=str, default='2d', help='output data directory')
    
    args = parser.parse_args()
    if args.mode not in ['2.5d', '3d', '2d']:
        raise RuntimeError("Unexpected map type", args.mode)

    data_dir = args.data_dir
    output_dir = args.output_dir or args.data_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    data = load_data(data_dir)

    img_data = load_images(data_dir)

    map_builder = MapBuilder(data['camera_matrix'],
                             data['tag_side_length'],
                             args.mode)


    viewpoint_ids = sorted(data["viewpoints"].keys())
    next_viewpoint_idx = 0
    need_add_viewpoint = True

    # for viewpoint_id, tags in data["viewpoints"].items():
    #     map_builder.add_viewpoint(viewpoint_id, tags)
    # map_builder.relinearize()
    print("Starting")

    prev_error = float('inf')
    while True:
        need_add_viewpoint = need_add_viewpoint or (args.mode == '2d' and next_viewpoint_idx + 1 <= len(viewpoint_ids))
        if need_add_viewpoint:
            viewpoint_id = viewpoint_ids[next_viewpoint_idx]
            tags = data["viewpoints"][viewpoint_id]
            map_builder.add_viewpoint(viewpoint_id, tags)
            print("Added viewpoint", viewpoint_id)
            map_builder.relinearize()
            need_add_viewpoint = False
            next_viewpoint_idx += 1            
        
        for i in range(20):
            map_builder.send_detection_to_viewpoint_msgs()
            map_builder.send_detection_to_tag_msgs()
        improved = map_builder.update()
        error = map_builder.get_total_detection_error()

        if prev_error != float('inf'):
            delta = error - prev_error
            change = delta/prev_error
            avg_det_error = error / len(map_builder.detections)

            print("tracking", next_viewpoint_idx, "viewpoints",
                  "error", error, "avg_error", avg_det_error, "change", change*100, "%")

            viewpoint_converged = improved and (abs(change) < 1e-5 or avg_det_error < 10)
            if viewpoint_converged:
                if next_viewpoint_idx+1 <= len(viewpoint_ids):
                    need_add_viewpoint = True
                elif abs(change) < 1e-6:
                    # no more viewpoints and converged
                    break

        prev_error = error

    print("Saving to", output_dir)
    save_viewpoints_json(
            output_dir,
            map_builder.viewpoint_ids,
            map_builder.txs_world_viewpoint)

    if args.mode == '3d':
        save_map3d_json(
            output_dir,
            map_builder.tag_side_length,
            map_builder.tag_ids,
            map_builder.txs_world_tag)
    elif args.mode == '2.5d':
        save_map2p5d_json(
            output_dir,
            map_builder.tag_side_length,
            map_builder.tag_ids,
            map_builder.txs_world_tag)
    else:
        save_map_json(
            output_dir,
            map_builder.tag_side_length,
            map_builder.tag_ids,
            map_builder.txs_world_tag)

if __name__ == "__main__":
    main()
    
