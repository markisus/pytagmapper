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

kTol = 1e-8

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

def so3_to_vector(so3):
    return np.array([[so3[2,1], so3[0,2], so3[1,0]]]).T

def se3_to_matrix(se3):
    result = np.empty((4,4))
    result[:3,:3] = so3_to_matrix(se3[:3,:])
    result[:3,3] = se3[3:,0]
    result[3,:] = 0
    return result

def se3_exp(se3):
    assert se3.shape == (6,1)
    # See page 10 https://ethaneade.com/lie.pdf
    # we reverse u and omega in the ordering of se3
    result = np.eye(4)
    omega_vec = se3[:3,:]
    theta_squared = np.dot(omega_vec.T, omega_vec)
    omega = so3_to_matrix(omega_vec)
    if (theta_squared < kTol): 
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

    if (abs(theta) < kTol):
        return np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])

    r_p = (x + 1j*y)/theta
    r = r_p * 1j
    rotation = np.cos(theta) + 1j*np.sin(theta)
    ep = r + (-r * rotation)
    ed = rotation

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

def almost_equals(a, b):
    return abs(a - b) < kTol

def SO3_log_decomposed(SO3):
    """
    Returns so3_hat, the normalized version of the so3
    vector corresponding to the input SO3, and theta, the
    magnitude.
    """
    trace = np.trace(SO3)
    is_identity = almost_equals(trace, 3)

    # Edge case: identity
    if is_identity:
        theta = 0
        omega_hat = np.array([[1, 0, 0]], dtype=float).T
        return omega_hat, theta

    # Edge case: rotation of k*PI
    if almost_equals(trace, -1):
        # print("Edge case")
        theta = np.pi
        r33 = SO3[2,2]
        r22 = SO3[1,1]
        r11 = SO3[0,0]

        if not almost_equals(1.0 + r33, 0):
            # print("Case 1")
            omega_hat = np.array([[SO3[0,2], SO3[1,2], 1+SO3[2,2]]]).T
            omega_hat /= (2 * (1 + r33))**0.5
            return omega_hat, theta

        if not almost_equals(1.0 + r22, 0):
            # print("Case 2")
            omega_hat = np.array([[SO3[0,1], 1+SO3[1,1], SO3[2,1]]]).T
            omega_hat /= (2 * (1 + r22))**0.5
            return omega_hat, theta

        # print("Case 3")
        assert almost_equals(1.0 + r33, 0)
        omega_hat = np.array([[1+SO3[0,0], SO3[1,0], SO3[2,0]]]).T
        omega_hat /= (2 * (1 + r11))**0.5
        return omega_hat, theta

    # normal case
    # htmo means Half of Trace Minus One
    htmo = 0.5 * (trace - 1)
    theta = np.arccos(htmo)
    sin_acos_htmo = (1.0 - htmo*htmo)**0.5
    omega_mat = 0.5/sin_acos_htmo * (SO3 - SO3.T)
    omega_hat = so3_to_vector(omega_mat)
    return omega_hat, theta

def SO3_log(SO3):
    so3_hat, theta = SO3_log_decomposed(SO3)
    return so3_hat * theta

def barfoot_Q(se3):
    rho = so3_to_matrix(se3[3:,:])
    theta = so3_to_matrix(se3[:3,:])
    rho_theta = (rho@theta)
    rho_theta2 = (rho_theta@theta)
    theta_rho_theta = (theta@rho_theta)
    theta_rho = (theta@rho)
    theta2_rho = (theta@theta_rho)

    angle = np.linalg.norm(se3[:3,:])
    if (angle < kTol):
        angle2 = angle*angle
        angle4 = angle2*angle2
        c1 = 1.0/6 - angle2/120 + angle4/5040
        c2 = 1.0/24 - angle2/720 + angle4/40320
        c3 = 1.0/120 - angle2/2520 + angle4/120960
    else:
        angle2 = angle*angle
        angle3 = angle2*angle
        angle4 = angle2*angle2
        angle5 = angle3*angle2
        sn = np.sin(angle)
        cs = np.cos(angle)
        c1 = (angle - sn)/angle3
        c2 = -(1.0 - angle2/2 - cs)/angle4
        c3 = -0.5*(
            (1.0 - angle2/2 - cs)/angle4 -
            3.0*(angle - sn - angle3/6)/angle5)

    line1 = 0.5*rho + c1*(theta_rho + rho_theta + theta_rho_theta)
    line2 = c2*(theta2_rho + rho_theta2 - 3.0*theta_rho_theta)
    line3 = c3*(theta_rho_theta@theta + theta@theta_rho_theta)
    return line1 + line2 + line3

def SO3_left_jacobian(so3):
    theta = np.linalg.norm(so3)
    theta2 = theta*theta
    theta3 = theta2*theta
    theta4 = theta2*theta2

    theta_mat = so3_to_matrix(so3)    
    if (theta < kTol):
        coeff1 = 0.5 - theta2/24 + theta4/720
        coeff2 = 1.0/6 - theta2/120 + theta4/5040
    else:
        cs = np.cos(theta)
        sn = np.sin(theta)
        coeff1 = (1.0 - cs)/theta2
        coeff2 = (theta - sn)/theta3

    result = np.eye(3, dtype=float) + coeff1*theta_mat + coeff2*theta_mat@theta_mat
    return result

def se3_left_jacobian(se3):
    SO3_lj = SO3_left_jacobian(se3[:3,:])
    Q = barfoot_Q(se3)
    result = np.empty((6,6), dtype=float)
    result[:3,:3] = SO3_lj
    result[3:,3:] = SO3_lj
    result[:3,3:] = 0
    result[3:,:3] = Q
    return result

def x_cotx(x):
    return x / np.tan(x)
    c2 = -1.0 / 3
    c4 = -1.0 / 45
    c6 = -2.0 / 945
    c8 = -1.0 / 4725
    c10 = -2.0 / 93555
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    return 1.0 + c2 * x2 + c4 * x4 + c6 * x6 + c8 * x8 + c10 * x10

def se3_to_vector(se3_mat):
    se3 = np.empty((6,1), dtype=float)
    se3[:3,:] = so3_to_vector(se3_mat[:3,:3])
    se3[3:,:] = se3_mat[:3,3:]
    return se3

def SE3_log(SE3):
    omega_hat, theta = SO3_log_decomposed(SE3[:3,:3])
    omega = so3_to_matrix(omega_hat)

    p = SE3[:3,3:]
    omega_p = omega @ p
    v_theta = p - 0.5*theta*omega_p + (1.0 - x_cotx(theta/2)) * omega @ omega_p
    result = np.empty((6,1), float)
    result[:3,:] = omega_hat * theta
    result[3:,:] = v_theta
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
    if abs(normRxy - 1) > kTol:
        print("Bad SE2\n", SE2)
        print("Rxy", Rxy.T)
        print("normRxy ", normRxy)
        raise RuntimeError("Bad SE2", SE2)
    
    if np.max(np.abs(R @ R.T - np.eye(2))) > kTol:
        print("Bad SE2\n", SE2)
        print("R@R.T\n", R@R.T)
        print("normRxy\n", normRxy)
        raise RuntimeError("Bad SE2", SE2)

    if abs(SE2[2,2] - 1) > kTol:
        print("Bad SE2\n", SE2)
        print("Lower corner not 1")
        raise RuntimeError("Bad SE2", SE2)

    if np.max(np.abs(SE2[2,:2])) > kTol:
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

    
    so3 = np.array([[0.2, -0.8, 0.1]]).T
    so3_matrix = so3_to_matrix(so3)
    SO3 = scipy.linalg.expm(so3_matrix)
    print("SO3 matrix")
    print(SO3)
    so3_hat, theta = SO3_log_decomposed(SO3)
    so3_result = so3_hat * theta
    print(so3_result, so3)
    print("SO3_result matrix")
    SO3_result = scipy.linalg.expm(so3_to_matrix(so3_result))
    print(SO3_result)

    # se3 = rng.random((6,1))
    # SE3 = se3_exp(se3)
    # se3_result = SE3_log(SE3)
    # print("se3, se3_result", se3.T, se3_result.T)

    # logM = se3
    # M = SE3
    # # exp(Jl_logM [Δ]) * M ~= exp(logM + [Δ])
    # # exp(Jl_logM [Δ]) ~= exp(logM + [Δ]) M_inv
    # # Jl_logM [Δ] ~= log(exp(logM + [Δ]) M_inv)
    # # Jl_logM [Δ] ~= lim_t->0 [log(exp(logM + [Δ t]) M_inv) / t]

    # for i in range(6):
    #     print("i == ", i)
    #     perturb = np.zeros((6,1), dtype=float)
    #     perturb[i,0] = 1
    #     eps = 1e-3
    #     expected = SE3_log(se3_exp(logM + perturb*eps) @ SE3_inv(M))/eps
    #     print("Expected ", expected.T)
    #     print("Actual ", (se3_left_jacobian(logM) @ perturb).T)


    # dxy_dobject = dxy_campoint * dcampoint_dobject
    # dobject is an se3 perturbation
    
    # dxy = dxy_dobject * dse3
    #     = dxy_dobject * log(tx_world_object2 * tx_world_object1.inv)
    #     = dxy_dobject * log(exp(se3_world_object1 + delta) * tx_world_object1.inv)
    #     = dxy_dobject * jl(se3_world_object1) * delta

    # round trip
    # SE3 = np.array([[ 0.99283688,  0.03183017,  0.11515975, -0.01986499],
    #     [-0.05648195,  0.97439577,  0.21762966, -0.0552138 ],
    #     [-0.10528398, -0.2225752 ,  0.96921389,  0.07555235],
    #     [ 0.0        ,  0.0        ,  0.0        ,  1.0        ]])

    # se3 = SE3_log(SE3)
    # se3_np = se3_to_vector(scipy.linalg.logm(SE3))

    # print("se3 ", se3.T)
    # print("se3 mat np ", se3_np.T)

    # SE3_round_trip = se3_exp(se3)

    # print("SE3\n", SE3)
    # print("SE3_round_trip\n", SE3_round_trip)

    SE3 = np.array([[ 1.0,   0.0,   0.0,   0.0 ],
                    [ 0.0,  -1.0,   0.0,   0.0 ],
                    [ 0.0,   0.0,  -1.0,   0.3],
                    [ 0.0,   0.0,   0.0,   1.0 ]])
    se3 = SE3_log(SE3)
    SE3_round_trip = se3_exp(se3)

    print("SE3\n", SE3)
    print("SE3_round_trip\n", SE3_round_trip)

    
    
