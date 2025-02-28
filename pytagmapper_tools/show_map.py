from hack_sys_path import *

from pytagmapper.data import *
from pytagmapper.geometry import *
from pytagmapper.project import *
import argparse
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

        plt.axis('scaled')
        plt.show()
    elif map_type == '3d':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for tag_id, pose_world_tag in map_data['tag_locations'].items():
            if tag_id in tag_side_lengths:
                tag_side_length = tag_side_lengths[tag_id]
            else:
                tag_side_length = tag_side_lengths["default"]
            
            # 4x4 transform of tag in "world" frame
            tx_world_tag = np.array(pose_world_tag)

            # 4xN matrix of corners in tag’s local frame
            tag_corners_3d = get_corners_mat(tag_side_length)  # shape: (4, N), typically N=4 or 5

            # Transform corners from tag frame to world frame
            world_corners = tx_world_tag @ tag_corners_3d  # shape: (4, N)

            # Plot each corner edge in 3D
            for i in range(4):
                x1, y1, z1, _ = world_corners[:, i]
                x2, y2, z2, _ = world_corners[:, (i + 1) % 4]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-')  # 'b-' for a blue line

            # Label the center of the tag
            center = np.mean(world_corners[:3, :4], axis=1)  # average of the four corners (x,y,z)
            ax.text(center[0], center[1], center[2], str(tag_id), color='red')

        # Make all axes have the same scale
        def set_axes_equal_3d(ax):
            """Make axes of 3D plot have equal scale so cubes look like cubes."""
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            x_range = x_limits[1] - x_limits[0]
            y_range = y_limits[1] - y_limits[0]
            z_range = z_limits[1] - z_limits[0]
            max_range = max(x_range, y_range, z_range)
            
            mid_x = 0.5 * (x_limits[0] + x_limits[1])
            mid_y = 0.5 * (y_limits[0] + y_limits[1])
            mid_z = 0.5 * (z_limits[0] + z_limits[1])
            
            ax.set_xlim3d([mid_x - max_range / 2, mid_x + max_range / 2])
            ax.set_ylim3d([mid_y - max_range / 2, mid_y + max_range / 2])
            ax.set_zlim3d([mid_z - max_range / 2, mid_z + max_range / 2])

        set_axes_equal_3d(ax)
        plt.show()
    else:
        raise RuntimeError("Unsupported map type", map_type)
        

if __name__ == "__main__":
    main()
    
