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
    parser.add_argument('--input_data_dir', type=str, help='input data directory')
    parser.add_argument('--output_data_dir', type=str, help='output data directory')
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

    # generate map viz
    tag_corners_2d = get_corners_mat2d(data['tag_side_length'])
    for tag_idx, tx_world_tag in enumerate(map_builder.txs_world_tag):
        tag_id = map_builder.tag_ids[tag_idx]
        world_corners = tx_world_tag @ tag_corners_2d
        for i in range(4):
            x1 = world_corners[0,i]
            x2 = world_corners[0,(i+1)%4]
            y1 = world_corners[1,i]
            y2 = world_corners[1,(i+1)%4]
            line = plt.Line2D((x1,x2), (y1,y2), lw=1.5)
            plt.gca().add_line(line)

        center = np.sum(world_corners, axis=1)/4
        plt.text(center[0], center[1], str(tag_id))
    plt.axis('scaled')
    plt.show()

if __name__ == "__main__":
    main()
    
