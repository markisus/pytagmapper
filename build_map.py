import argparse
import math
from map_builder import MapBuilder
from data import *
from geometry import *
import numpy as np
import matplotlib.pyplot as plt
from project import *

def main():
    parser = argparse.ArgumentParser(description='Build a map from images of tags.')
    parser.add_argument('--input-data-dir', type=str, help='input data directory')
    parser.add_argument('--output-data-dir', type=str, help='output data directory')
    args = parser.parse_args()
    
    data_dir = args.input_data_dir
    data_out = args.output_data_dir

    if not os.path.exists(data_out):
        os.mkdir(data_out)
    
    data = load_data(data_dir)

    img_data = load_images(data_dir)
    map_builder = MapBuilder(data['camera_matrix'],
                             data['tag_side_length'])

    for viewpoint_id, tags in data["viewpoints"].items():
        map_builder.add_viewpoint(viewpoint_id, tags)
    map_builder.relinearize()

    prev_error = float('inf')
    for j in range(100):
        for i in range(60):
            map_builder.send_detection_to_viewpoint_msgs()
            map_builder.send_detection_to_tag_msgs()
        map_builder.update()
        error = map_builder.get_total_detection_error()

        if prev_error != float('inf'):
            delta = error - prev_error
            change = delta/prev_error
            print("iteration", j, "error", error, "change", change*100, "%")
            if abs(change) < 1e-6:
                break

        prev_error = error


    print("Saving to", data_out)
    save_map_json(
        data_out,
        map_builder.tag_side_length,
        map_builder.tag_ids,
        map_builder.txs_world_tag)

    save_viewpoints_json(
        data_out,
        map_builder.viewpoint_ids,
        map_builder.txs_world_viewpoint)

if __name__ == "__main__":
    main()
    
