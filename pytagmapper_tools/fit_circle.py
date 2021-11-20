# MIT License

# Copyright (c) 2021 Mark Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np

def fit_circle(points):
    # require len(points) >= 3

    # circle fitting coope method
    # 2 xc xi + 2 yc yi + c = xi**2 + yi**2
    # and c + xc**2 + yc**2 = r**2
    rows = len(points)
    problem_mat = np.empty((rows, 3))
    soln_vec = np.empty((rows,1))
    for i, (x,y) in enumerate(points):
        problem_mat[i, :] = [2*x, 2*y, 1]
        soln_vec[i, :] = x**2 + y**2
    result = np.linalg.lstsq(problem_mat, soln_vec, rcond=None)
    circle_params = result[0]
    cx = circle_params[0,0]
    cy = circle_params[1,0]
    c = circle_params[2,0]
    cr = (cx**2 + cy**2 + c)**0.5
    return cx, cy, cr
