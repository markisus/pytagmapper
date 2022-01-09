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

from collections import namedtuple
import imgui

OverlayableCtx = namedtuple(
    "OverlayableCtx",
    [
        "data_width",
        "data_height",
        "display_width",
        "display_height",
        "corner_x",
        "corner_y",
        "scale"
    ])

def draw_overlayable_image(texture_id, data_width, data_height, display_width = 0):
    if not display_width:
        display_width = data_width
    scale = display_width / data_width
    display_height = scale * data_height
    screen_pos = imgui.get_cursor_screen_pos()
    imgui.image(texture_id, display_width, display_height)
    return OverlayableCtx(
        data_width,
        data_height,
        display_width,
        display_height,
        screen_pos.x,
        screen_pos.y,
        scale)

def draw_overlayable_rectangle(data_width, data_height, display_width = 0):
    if not display_width:
        display_width = data_width
    scale = display_width / data_width
    display_height = scale * data_height
    screen_pos = imgui.get_cursor_screen_pos()
    imgui.invisible_button("", display_width, display_height)
    return OverlayableCtx(
        data_width,
        data_height,
        display_width,
        display_height,
        screen_pos.x,
        screen_pos.y,
        scale)

def overlay_transform(overlayable, x, y):
    sx = overlayable.scale * x + overlayable.corner_x
    sy = overlayable.scale * y + overlayable.corner_y
    return sx, sy

def overlay_inv_transform(overlayable, sx, sy):
    x = (sx - overlayable.corner_x) / overlayable.scale
    y = (sy - overlayable.corner_y) / overlayable.scale
    return x, y

def is_oob(overlayable, x, y):
    if x <= 0:
        return True
    if y <= 0:
        return True
    if x > overlayable.data_width:
        return True
    if y > overlayable.data_height:
        return True
    return False

def overlay_line(overlayable, x1, y1, x2, y2, color, thickness):
    # poor man's clip rect until clip rect is implemented in pyimgui
    if is_oob(overlayable, x1, y1):
        return
    if is_oob(overlayable, x2, y2):
        return
    
    sx1, sy1 = overlay_transform(overlayable, x1, y1)
    sx2, sy2 = overlay_transform(overlayable, x2, y2)

    imgui.get_window_draw_list().add_line(
        sx1, sy1, sx2, sy2, color, thickness)

def overlay_circle(overlayable, x, y, r, color, thickness, num_segments=12):
    # poor man's clip rect until clip rect is implemented in pyimgui
    if is_oob(overlayable, x, y):
        return
    
    sx, sy = overlay_transform(overlayable, x, y)
    sr = overlayable.scale * r
    imgui.get_window_draw_list().add_circle(
        sx, sy, sr, col=color, num_segments=num_segments, thickness=thickness)

def overlay_circle_filled(overlayable, x, y, r, color, num_segments=12):
    # poor man's clip rect until clip rect is implemented in pyimgui
    if is_oob(overlayable, x, y):
        return
    
    sx, sy = overlay_transform(overlayable, x, y)
    sr = overlayable.scale * r
    imgui.get_window_draw_list().add_circle_filled(
        sx, sy, sr, col=color, num_segments=num_segments)
    

def overlay_text(overlayable, x, y, color, text):
    # poor man's clip rect until clip rect is implemented in pyimgui
    if is_oob(overlayable, x, y):
        return
    
    sx, sy = overlay_transform(overlayable, x, y)
    imgui.get_window_draw_list().add_text(sx, sy, color, text)

def overlay_polyline(overlayable, polyline, colors, thickness):
    for i in range(polyline.shape[1]):
        ni = (i + 1) % polyline.shape[1]
        overlay_line(overlayable,
                     polyline[0,i],
                     polyline[1,i],
                     polyline[0,ni],
                     polyline[1,ni],
                     colors[i],
                     thickness)
    
