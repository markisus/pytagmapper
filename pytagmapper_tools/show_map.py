from hack_sys_path import *

from pytagmapper.data import *
from pytagmapper.geometry import *
from pytagmapper.project import *
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Plot a tag map.')
    parser.add_argument('map_dir', type=str, help='map directory')
    args = parser.parse_args()

    map_data = load_map(args.map_dir)
    tag_side_lengths = map_data['tag_side_lengths']

    map_type = map_data['map_type']

    # 2 and 2.5d maps
    if map_type == '2.5d' or map_type == '2d':
        for tag_id, pose_world_tag in map_data['tag_locations'].items():
            if tag_id in tag_side_lengths:
                tag_side_length = tag_side_lengths[tag_id]
            else:
                tag_side_length = tag_side_lengths["default"]

            tag_corners_2d = get_corners_mat2d(tag_side_length)
                
            if map_type == '2.5d':
                xyt_world_tag = pose_world_tag[:3]
            else:
                xyt_world_tag = pose_world_tag

            tx_world_tag = xyt_to_SE2(np.array([xyt_world_tag]).T)
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

            if map_type == '2.5d':
                z = "{:#.4g}".format(pose_world_tag[3])
                plt.text(center[0], center[1] - tag_side_length/2, f"z={z}")
    elif map_type == '3d':
        for tag_id, pose_world_tag in map_data['tag_locations'].items():
            if tag_id in tag_side_lengths:
                tag_side_length = tag_side_lengths[tag_id]
            else:
                tag_side_length = tag_side_lengths["default"]
            tag_corners_3d = get_corners_mat(tag_side_length)                
            
            tx_world_tag = np.array(pose_world_tag)
            world_corners = tx_world_tag @ tag_corners_3d
            for i in range(4):
                x1 = world_corners[0,i]
                x2 = world_corners[0,(i+1)%4]
                y1 = world_corners[1,i]
                y2 = world_corners[1,(i+1)%4]
                line = plt.Line2D((x1,x2), (y1,y2), lw=1.5)
                plt.gca().add_line(line)

            center = np.sum(world_corners, axis=1)/4
            plt.text(center[0], center[1], str(tag_id))

            z = "{:#.4g}".format(tx_world_tag[2,3])
            plt.text(center[0], center[1] - tag_side_length/2, f"z={z}")
    else:
        raise RuntimeError("Unsupported map type", map_type)
        

    plt.axis('scaled')
    plt.show()

if __name__ == "__main__":
    main()
    
