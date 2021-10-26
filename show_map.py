import argparse
import math
from map_builder import MapBuilder
from data import *
from geometry import *
import numpy as np
import matplotlib.pyplot as plt
from project import *

def main():
    parser = argparse.ArgumentParser(description='Plot a tag map.')
    parser.add_argument('--map_dir', type=str, help='map directory')
    args = parser.parse_args()

    map_data = load_map(args.map_dir)
    tag_corners_2d = get_corners_mat2d(map_data['tag_side_length'])
    for tag_id, xyt_world_tag in map_data['tag_locations'].items():
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
    plt.axis('scaled')
    plt.show()

if __name__ == "__main__":
    main()
    
