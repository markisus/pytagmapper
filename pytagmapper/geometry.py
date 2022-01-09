# The following license applies to this particular file.
########################################################################
# This is free and unencumbered software released into the public domain.

# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.

# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <http://unlicense.org/>
########################################################################

import numpy as np
import math

def SE2_to_SE3(SE2):
    SE3 = np.eye(4)
    SE3[:2,:2] = SE2[:2,:2]
    SE3[:2,3] = SE2[:2,2]
    return SE3

def SE3_to_SE2(SE3):
    """try to coerce an SE3 into an SE2
    by assuming it's mostly flat in xy plane
    """
    SE2 = np.eye(3, dtype=np.float64)
    SE2[:2,:2] = SE3[:2,:2]
    SE2[:2,2:3] = SE3[:2,3:4]
    fix_SE2(SE2)
    return SE2

def xyt_to_SE3(xyt):
    x = xyt[0,0]
    y = xyt[1,0]
    t = xyt[2,0]
    ct = math.cos(t)
    st = math.sin(t)
    return np.array([
        [ct, -st, 0, x],
        [st,  ct, 0, y],
        [ 0,   0, 1, 0],
        [ 0,   0, 0, 1],
    ])

def xytz_to_SE3(xytz):
    x = xytz[0,0]
    y = xytz[1,0]
    t = xytz[2,0]
    z = xytz[3,0]
    ct = math.cos(t)
    st = math.sin(t)
    return np.array([
        [ct, -st, 0, x],
        [st,  ct, 0, y],
        [ 0,   0, 1, z],
        [ 0,   0, 0, 1],
    ])

def xyt_to_SE2(xyt):
    x = xyt[0,0]
    y = xyt[1,0]
    t = xyt[2,0]
    ct = math.cos(t)
    st = math.sin(t)
    return np.array([
        [ct, -st,  x],
        [st,  ct,  y],
        [ 0,   0,  1],
    ])

def so3_to_matrix(so3):
    wx = so3[0,0]
    wy = so3[1,0]
    wz = so3[2,0]
    so3_matrix = np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])
    return so3_matrix

def se3_to_matrix(se3):
    result = np.empty((4,4))
    result[:3,:3] = so3_to_matrix(se3[:3,:])
    result[:3,3] = se3[3:,0]
    result[3,:] = 0
    return result

def se3_exp(se3):
    # See page 10 https://ethaneade.com/lie.pdf
    # we reverse u and omega in the ordering of se3
    result = np.eye(4)
    omega_vec = se3[:3,:]
    theta_squared = np.dot(omega_vec.T, omega_vec)
    omega = so3_to_matrix(omega_vec)
    if (theta_squared < 1e-8): 
        # second order taylor expansion
        A = -theta_squared/6.0 + 1.0
        B = -theta_squared/24.0 + 0.5
        C = -theta_squared/120.0 + (1.0/6.0)
    else:
        theta = math.sqrt(theta_squared)
        stheta = math.sin(theta)
        ctheta = math.cos(theta)
        A = stheta / theta
        B = (-ctheta + 1.0) / theta_squared
        C = (-A + 1.0) / theta_squared

    omega_squared = omega @ omega
    v = se3[3:,:]

    result[:3,:3] += omega*A + omega_squared*B
    result[:3,3:4] = ((omega*B + omega_squared*C) @ v) + v
    return result

def se2_exp(se2):
    theta = se2[0,0]
    x = se2[1,0]
    y = se2[2,0]

    if (abs(theta) < 1e-6):
        return np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])

    r_p = (x + 1j*y)/theta
    r = r_p * 1j
    rotation = np.cos(theta) + 1j*np.sin(theta)
    ep = r + (-r * rotation);
    ed = rotation;

    return np.array([
        [ed.real, -ed.imag, ep.real],
        [ed.imag,  ed.real, ep.imag],
        [0,        0,       1      ]
    ])

    return out

def xyt_right_apply_se2(xyt, se2):
    SE2 = xyt_to_SE2(xyt)
    SE2 = SE2 @ se2_exp(se2)
    result = np.empty((3,1))
    result[0,0] = SE2[0,2]
    result[1,0] = SE2[1,2]
    result[2,0] = xyt[2,0] + se2[0,0]
    return result

def SE3_adj(SE3):
    result = np.empty((6,6))
    pm = so3_to_matrix(SE3[:3,3:4])
    result[:3,:3] = SE3[:3,:3]
    result[3:,3:] = SE3[:3,:3]
    result[3:,:3] = pm @ SE3[:3,:3]
    result[:3,3:].fill(0)
    return result

def SE3_inv(SE3):
    result = np.empty((4,4))
    result[:3,:3] = SE3[:3,:3].T
    result[:3,3:4] = -SE3[:3,:3].T @ SE3[:3,3:4]
    result[3,:] = [0, 0, 0, 1]
    return result

def fix_SE3(SE3):
    Rx = SE3[:3, 0]
    Ry = SE3[:3, 1]
    Rz = SE3[:3, 2]

    Rz = np.cross(Rx, Ry)
    Ry = np.cross(Rz, Rx)

    Rx /= np.linalg.norm(Rx)
    Ry /= np.linalg.norm(Ry)
    Rz /= np.linalg.norm(Rz)

    SE3[:3, 0] = Rx
    SE3[:3, 1] = Ry
    SE3[:3, 2] = Rz

    SE3[3, :] = [0,0,0,1]

def SE2_inv(SE2):
    result = np.empty((3,3))
    result[:2,:2] = SE2[:2,:2].T
    result[:2,2:3] = -SE2[:2,:2].T @ SE2[:2,2:3]
    result[2,:] = [0, 0, 1]
    return result

def fix_SE2(SE2):
    SE2[0,1] = -SE2[1,0]
    SE2[1,1] = SE2[0,0]
    Rxy = SE2[:2,0]
    normRxy = np.linalg.norm(Rxy)
    SE2[:2,:2] /= normRxy

def check_SE2(SE2):
    R = SE2[:2,:2]
    
    Rxy = R[:2,0]
    normRxy = np.linalg.norm(Rxy)
    if abs(normRxy - 1) > 1e-6:
        print("Bad SE2\n", SE2)
        print("Rxy", Rxy.T)
        print("normRxy ", normRxy)
        raise RuntimeError("Bad SE2", SE2)
    
    if np.max(np.abs(R @ R.T - np.eye(2))) > 1e-4:
        print("Bad SE2\n", SE2)
        print("R@R.T\n", R@R.T)
        print("normRxy\n", normRxy)
        raise RuntimeError("Bad SE2", SE2)

    if abs(SE2[2,2] - 1) > 1e-6:
        print("Bad SE2\n", SE2)
        print("Lower corner not 1")
        raise RuntimeError("Bad SE2", SE2)

    if np.max(np.abs(SE2[2,:2])) > 1e-6:
        print("Bad SE2\n", SE2)
        print("Lower zeros")
        raise RuntimeError("Bad SE2", SE2)

if __name__ == "__main__":
    import scipy.linalg
    rng = np.random.default_rng(0)
    se3 = rng.random((6,1))
    SE3 = se3_exp(se3)
    print(scipy.linalg.expm(se3_to_matrix(se3)))

    se2 = rng.random((3,1))
    print(se2_exp(se2))

    perturb = rng.random((6,1))
    epsilon = 1e-5
    # SE3 * exp(perturb) - SE3 = exp(adj perturb) - I

    SE3p = SE3 @ se3_exp(perturb*epsilon)
    expected = (SE3p - SE3)/epsilon @ SE3_inv(SE3)
    body_twist = SE3_inv(SE3) @ (SE3p - SE3)/epsilon
    print("bodytwist", body_twist)
    actual = SE3_adj(SE3) @ perturb
    print("exp", expected)
    print("act", se3_to_matrix(actual))

    print("toSE3", SE2_to_SE3(se2_exp(se2)))

    xyt = rng.random((3,1))
    xyt_update = xyt_right_apply_se2(xyt, se2)

    SE2_a = xyt_to_SE2(xyt)
    SE2_b = xyt_to_SE2(xyt_update)
    SE2_c = SE2_a @ se2_exp(se2)

    print("right apply se2 check", SE2_c - SE2_b)

    # check inv
    print("SE2 inv check\n", SE2_a @ SE2_inv(SE2_a))

    check_SE2(SE2_a)
    check_SE2(SE2_b)
    check_SE2(SE2_c)
    try:
        check_SE2(np.zeros((3,3)))
    except:
        print("check_SE2 failed as expected")

    # check se2 exp vs se3 exp
    se2 = rng.random((3,1))
    se3_from_se2 = np.zeros((6,1))
    se3_from_se2[2:5,:] = se2
    expected = se3_exp(se3_from_se2)
    actual = SE2_to_SE3(se2_exp(se2))
    print("expected", expected)
    print("se2 exp vs se3 exp\n", expected-actual)

    # check se2 wz
    se2 = np.array([[1, 0, 0]]).T
    print(se2_exp(se2))

    SE3 = np.array([
        [0.59, 0.80, 0,     0],
        [1.3, 1.80, 0, -0.01],
        [0, 0, 1, -0.01],
        [0, 0, 0,     1],
    ])

    fix_SE3(SE3)
    print("Fixed SE3\n", SE3)
