import json
import glob
import os
import cv2
import math
import numpy as np

def get_path(data_dir, filename):
    return os.path.join(data_dir, filename)

def load_measurements(data_dir):
    measurements = []
    with open(get_path(data_dir, "measurements.txt"), "r") as f:
        for line in f.readlines():
            tokens = line.split(" ")
            tag_a = int(tokens[0])
            tag_b = int(tokens[1])
            dist = float(tokens[2])
            measurements.append((tag_a, tag_b, dist))
    return measurements

def get_map_json(tag_side_length, tag_ids, txs_world_tag):
    map_data = {
        'tag_side_length': tag_side_length,
        'tag_locations': {},
        'map_type': '2d'
    }
    
    for tag_idx, tx_world_tag in enumerate(txs_world_tag):
        # convert to xyt
        x_dir = tx_world_tag[0,0]
        y_dir = tx_world_tag[1,0]
        theta = math.atan2(y_dir, x_dir)
        x = tx_world_tag[0,2]
        y = tx_world_tag[1,2]
        tag_id = tag_ids[tag_idx]
        map_data['tag_locations'][tag_id] = [x, y, theta]

    return map_data

def get_map2p5d_json(tag_side_length, tag_ids, txs_world_tag):
    map_data = {
        'tag_side_length': tag_side_length,
        'tag_locations': {},
        'map_type': '2.5d'
    }
    
    for tag_idx, tx_world_tag in enumerate(txs_world_tag):
        # convert to xytz
        x_dir = tx_world_tag[0,0]
        y_dir = tx_world_tag[1,0]
        theta = math.atan2(y_dir, x_dir)
        x = tx_world_tag[0,3]
        y = tx_world_tag[1,3]
        z = tx_world_tag[2,3]
        tag_id = tag_ids[tag_idx]
        map_data['tag_locations'][tag_id] = [x, y, theta, z]

    return map_data


def save_map_json(data_dir, tag_side_length, tag_ids, txs_world_tag):
    with open(get_path(data_dir, "map.json"), "w") as f:
        json.dump(get_map_json(tag_side_length,
                               tag_ids,
                               txs_world_tag), f)

def save_map2p5d_json(data_dir, tag_side_length, tag_ids, txs_world_tag):
    with open(get_path(data_dir, "map.json"), "w") as f:
        json.dump(get_map2p5d_json(tag_side_length,
                                   tag_ids,
                                   txs_world_tag), f)

def save_viewpoints_json(data_dir, viewpoint_ids, txs_world_viewpoint):
    data = {}
    for viewpoint_id, tx_world_viewpoint in zip(viewpoint_ids, txs_world_viewpoint):
        data[viewpoint_id] = tx_world_viewpoint.tolist()
    
    with open(get_path(data_dir, "viewpoints.json"), "w") as f:
        json.dump(data, f)
        
def load_map(data_dir):
    with open(get_path(data_dir, "map.json")) as f:
        data = json.load(f)

        # convert string keys to int keys
        tag_locations_fixed = {}
        for k, v in data['tag_locations'].items():
            tag_locations_fixed[int(k)] = v
        
        data['tag_locations'] = tag_locations_fixed
        return data

def load_viewpoints(data_dir):
    with open(get_viewpoints_json_path(data_dir)) as f:
        data = json.load(f)

        # convert string keys to int keys
        data_fixed = {}
        for k, v in data.items():
            data_fixed[int(k)] = np.array(v)

        return data_fixed

def parse_camera_matrix_file(f):
    camera_matrix = []
    for line in f:
        camera_matrix.append([float(s) for s in line.split(" ")])
    return np.array(camera_matrix)

def load_camera_matrix(data_dir = "data"):
    with open(get_path(data_dir, "camera_matrix.txt"), "r") as f:
        return parse_camera_matrix_file(f)

def load_tag_side_length(data_dir = "data"):
    with open(get_path(data_dir, "tag_side_length.txt"), "r") as f:
        return float(f.readline())

def load_data(data_dir = "data"):
    data = {
        'viewpoints': {},
        'camera_matrix': None,
        'tag_side_length': None
    }

    data['camera_matrix'] = load_camera_matrix(data_dir)
    data['tag_side_length'] = load_tag_side_length(data_dir)

    for file_path in glob.glob(os.path.join(data_dir, "tags_*.txt")):
        file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
        file_id = int(file_name.split("_")[-1])
        with open(file_path, "r") as f:
            data['viewpoints'][file_id] = parse_tag_file(f)

    return data

def get_image_paths(data_dir = "data"):
    image_paths = {}
    for file_path in glob.glob(os.path.join(data_dir, "image_*.png")):
        file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
        file_id = int(file_name.split("_")[-1])
        image_paths[file_id] = file_path
    return image_paths

def load_images(data_dir = "data"):
    data = {}
    for file_id, file_path in get_image_paths(data_dir).items():
        data[file_id] = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    return data

def parse_tag_file(f):
    tags = {}
    current_tag = None
    for idx, line in enumerate(f.readlines()):
        if idx % 5 == 0:
            tag_id = int(line)
            tags[tag_id] = []
            current_tag = tags[tag_id]
        else:
            current_tag += [float(s) for s in line.split(" ")]
    return tags
