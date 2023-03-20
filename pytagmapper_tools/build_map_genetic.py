from hack_sys_path import *
import argparse
import random
from pytagmapper import data
from pytagmapper.geometry import *
from pytagmapper.map_builder import MapBuilder
import sys
import copy

import cv2
import numpy as np

rng = np.random.default_rng(0)

def look_at_origin(from_xyz, up_dir):
    from_xyz = np.array(from_xyz, dtype=np.float64)
    up_dir = np.array(up_dir, dtype=np.float64)
    look_dir = -from_xyz
    look_dir /= np.linalg.norm(look_dir)
    x = np.cross(look_dir, up_dir) # z cross (-y) = x
    y = np.cross(look_dir, x) # z cross x = y
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    result = np.empty((4,4))
    result[:3,0] = x
    result[:3,1] = y
    result[:3,2] = look_dir
    result[:3,3] = from_xyz
    result[3,:] = [0, 0, 0, 1]
    return result

# success of the tracker heavily depends on initialization
# initialization from one of these viewpoints generally will succeed
INIT_TXS_WORLD_VIEWPOINT = [
    # topdown views
    look_at_origin([0,0,1], [0,1,0]),
    look_at_origin([0,0,1], [0,-1,0]),
    look_at_origin([0,0,1], [1,0,0]),
    look_at_origin([0,0,1], [-1,0,0]),

    # view from left
    look_at_origin([1,0,0.5], [0,0,1]),
    look_at_origin([1,0,0.5], [0,0,-1]),
    look_at_origin([1,0,0.5], [0,1,0]),
    look_at_origin([1,0,0.5], [0,-1,0]),

    # view from top
    look_at_origin([0,1,0.5], [0,0,1]),
    look_at_origin([0,1,0.5], [0,0,-1]),
    look_at_origin([0,1,0.5], [1,0,0]),
    look_at_origin([0,1,0.5], [-1,0,0]),

    # view from right
    look_at_origin([-1,0,0.5], [0,0,1]),
    look_at_origin([-1,0,0.5], [0,0,-1]),
    look_at_origin([-1,0,0.5], [0,1,0]),
    look_at_origin([-1,0,0.5], [0,-1,0]),

    # view from bottom
    look_at_origin([0,-1,0.5], [0,0,1]),
    look_at_origin([0,-1,0.5], [0,0,-1]),
    look_at_origin([0,-1,0.5], [1,0,0]),
    look_at_origin([0,-1,0.5], [-1,0,0]),
]

def solvePnPWrapper(obj_points, img_points, camera_matrix):
    succ, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, None)
    if not succ:
        raise RuntimeError("solvePnP failed")
    rot, _ = cv2.Rodrigues(rvec)
    tx_camera_obj = np.eye(4, dtype=np.float64)
    tx_camera_obj[:3,:3] = rot
    tx_camera_obj[:3,3:4] = tvec
    return tx_camera_obj

def pop_random(l):
    rand_idx = random.randint(0, len(l)-1)
    selected = l[rand_idx]
    l[rand_idx] = l[-1]
    l.pop()
    return selected

def eval_fitness(organism):
    detection_errors = organism['map_builder'].detection_errors
    fitness = 0
    for error in detection_errors:
        fitness -= error
        if error < 5:
            # point for good overlap
            fitness += 10
    organism['fitness'] = fitness
    return fitness
            
def init_random_genome(scene_data):
    # viewpoint_id, tag_id
    unused_detections = []
    for viewpoint_id, viewpoint in scene_data['viewpoints'].items():
        for tag_id in viewpoint.keys():
            unused_detections.append((viewpoint_id, tag_id))
    
    camera_matrix = scene_data['camera_matrix']
    viewpoints = scene_data['viewpoints']

    viewpoint_id, tag_id = pop_random(unused_detections)

    map_builder = MapBuilder(scene_data['camera_matrix'],
                             scene_data['tag_side_lengths'],
                             '3d')

    # pretend the tag is at the origin
    # then use one of the heuristic initializers
    tx_tag_viewpoint = random.choice(INIT_TXS_WORLD_VIEWPOINT)
    tx_tag_viewpoint[:3,3] *= random.uniform(0.01, 1.0)
    viewpoint_perturb = rng.uniform(-1, 1, (6,1))
    viewpoint_perturb[:3,:] *= np.pi/10
    viewpoint_perturb[3:,:] *= 0.1
    tx_tag_viewpoint = tx_tag_viewpoint @ se3_exp(viewpoint_perturb)

    se3_world_tag = rng.uniform(-1, 1, (6,1))
    se3_world_tag[:3,:] *= 2*np.pi
    se3_world_tag[3:,:] *= 1.0
    tx_world_tag = np.eye(4, dtype=float)
    tx_world_viewpoint = tx_world_tag @ tx_tag_viewpoint
    
    map_builder.init_viewpoint(viewpoint_id, tx_world_viewpoint)

    theta = random.uniform(-2 * np.pi, 2 * np.pi)
    height = random.uniform(0.01, 0.5)
    tx_viewpoint_tag = SE3_inv(tx_tag_viewpoint)

    map_builder.init_tag(tag_id, tx_world_tag)
    map_builder.init_detection(viewpoint_id, tag_id, scene_data['viewpoints'][viewpoint_id][tag_id])

    return {
        'age': 0,
        'map_builder': map_builder,
        'fitness': 0
    }

def mutate(organism, scene_data):
    builder = organism['map_builder']

    if random.uniform(0, 1) <= 0.15:
        # pick a random detection
    
    for it in range(2):
        builder.prioritized_update()
    organism['age'] += 1

def crossover(mother, father):
    mother_mb = mother['map_builder']
    father_mb = father['map_builder']
    
    child = {
        'age': max(mother['age'], father['age']),
        'map_builder': copy.deepcopy(mother['map_builder']),
        'fitness': 0
    }

    child_mb = child['map_builder']

    viewpoint_idx = random.randint(0, len(father_mb.viewpoint_ids)-1)
    viewpoint_id = father_mb.viewpoint_ids[viewpoint_idx]
    tx_world_viewpoint = father_mb.txs_world_viewpoint[viewpoint_idx]

    new_viewpoint = False
    if viewpoint_id not in child_mb.viewpoint_id_to_idx:
        new_viewpoint = True
        child_mb.init_viewpoint(viewpoint_id, tx_world_viewpoint)
    else:
        child_viewpoint_idx = child_mb.viewpoint_id_to_idx[viewpoint_id]
        tx_childworld_viewpoint = child_mb.txs_world_viewpoint[child_viewpoint_idx]

    for det_idx in father_mb.viewpoint_detections[viewpoint_idx]:
        if random.uniform(0, 1) <= 0.2:
            continue

        tag_idx, viewpoint_idx, tag_corners = father_mb.detections[det_idx]
        tag_id = father_mb.tag_ids[tag_idx]
        tx_world_tag = father_mb.txs_world_tag[tag_idx]

        new_tag = False
        if tag_id not in child_mb.tag_id_to_idx:
            new_tag = True
            child_mb.init_tag(tag_id, tx_world_tag)
        else:
            child_mb.reset_tag(tag_id, tx_world_tag)

        if new_tag or new_viewpoint:
            # then we also need to add the actual detection
            child_mb.init_detection(viewpoint_id, tag_id, tag_corners)
    
    return child

def does_b_dominate_a(a, b):
    # returns if a dominated by b
    return b['age'] <= a['age'] and b['fitness'] >= a['fitness']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic map builder")
    parser.add_argument('directory', type=str, help='scene data directory')
    parser.add_argument('--output-dir', '-o', type=str, help='output data directory', default='')
    args = parser.parse_args()

    output_dir = args.output_dir or args.directory
    scene_data = data.load_data(args.directory)

    population_cap = 100
    pool = []

    # init the population
    for i in range(population_cap):
        organism = init_random_genome(scene_data)
        mutate(organism, scene_data)
        pool.append(organism)

    while True:
        for organism in pool:
            mutate(organism, scene_data)
            eval_fitness(organism)

        new_organism = init_random_genome(scene_data)
        mutate(new_organism, scene_data)
        eval_fitness(new_organism)

        new_pool = [new_organism]

        while len(new_pool) < population_cap*0.5 and pool:
            # print("popping random from pool ", len(pool), " new pool ", len(new_pool))
            organism_a = pop_random(pool)
            if len(pool) == 0:
                new_pool.append(organism_a)
                break

            organism_b = pop_random(pool)

            if does_b_dominate_a(organism_a, organism_b):
                new_pool.append(organism_b)
            elif does_b_dominate_a(organism_b, organism_a):
                new_pool.append(organism_a)
            else:
                new_pool.append(organism_a)
                new_pool.append(organism_b)

        # reproduce
        while len(new_pool) < population_cap:
            # select random mother and father
            mother = random.choice(pool)
            father = random.choice(pool)
            child = crossover(mother, father)
            mutate(child, scene_data)
            eval_fitness(child)
            new_pool.append(child)

        pool = new_pool

        print([(o['age'], o['fitness']) for o in pool[:10]])


    inits.sort()
    for i in inits:
        print(i)
    
    # gene_pool = []
    

    # # a scene genome
    # #
    # # a list of
    # # se3_world_tag
    # # se3_world_viewpoint
    # #
    # # then unused viewpoints and tags
    # # 

    # viewpoints = scene_data['viewpoints']
    # viewpoint_ids = list(viewpoints.keys())
    # random.shuffle(viewpoint_ids)
    # # print(f"viewpoint ids {viewpoint_ids}")

    # used_viewpoints = set()

    # # get the image with the most tags
    # best_viewpoint = 0
    # best_num_tags = -1
    # for viewpoint_id in viewpoint_ids:
    #     if len(viewpoints[viewpoint_id]) > best_num_tags:
    #         best_num_tags = len(viewpoints[viewpoint_id])
    #         best_viewpoint = viewpoint_id

    # # print("best viewpoint was ", best_viewpoint)
    # # print("best num tags ", best_num_tags)

    # map_builder = MapBuilder(scene_data['camera_matrix'],
    #                          scene_data['tag_side_lengths'],
    #                          args.mode)

