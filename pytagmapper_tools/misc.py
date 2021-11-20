import numpy as np
import math # for math.pi

def quad_contains_pt(cwise_quad, pt):
    pt = np.array(pt).flatten()
    for i in range(4):
        ni = (i+1)%4
        v = cwise_quad[:2,i]
        nv = cwise_quad[:2,ni]
        direction = nv - v
        direction_perp = np.array([direction[1], -direction[0]])
        in_hp = np.dot(direction_perp, pt - v) < 0
        if not in_hp:
            return False
    return True

def line_near_pt(px, py, qx, qy, x, y):
    line_start = np.array([px, py])
    line_end = np.array([qx, qy])
    pt = np.array([x, y])
    line_dir = line_end - line_start
    line_length = np.linalg.norm(line_dir)
    line_dir /= line_length + 1e-6
    line_dir_perp = np.array([-line_dir[1], line_dir[0]])
    ydist = abs(np.dot(line_dir_perp, pt - line_start))
    xdist = np.dot(line_dir, pt - line_start)
    tol = 15
    return ydist < tol and -tol < xdist < line_length + tol

def make_ngon(x, y, radius, num_segments):
    theta = 2 * math.pi / num_segments
    rot = math.cos(theta) + 1j * math.sin(theta)
    current_rotor = 1.0 + 0j
    vertices = np.empty((4, num_segments))
    for i in range(num_segments):
        vertices[0,i] = radius * current_rotor.real + x
        vertices[1,i] = radius * current_rotor.imag + y
        vertices[2,i] = 0
        vertices[3,i] = 1
        current_rotor *= rot
    return vertices
        
