from hack_sys_path import *

from gl_util import GlRgbTexture
from imgui_sdl_wrapper import ImguiSdlWrapper
from overlayable import *
from fit_circle import *
from pytagmapper.data import *
from pytagmapper.project import *
from pytagmapper.geometry import *
from pytagmapper.inside_out_tracker import InsideOutTracker
from pytagmapper.geometry import SE3_inv
from rectified_tag_view import *
from misc import *
import argparse
import cv2
import imgui
import numpy as np
import os
import math
from collections import defaultdict

def project_points(camera_matrix, tx_camera_tag, xys):
    pts = np.empty((4,len(xys)))
    for i in range(len(xys)):
        x, y = xys[i]
        pts[:,i] = [x, y, 0, 1]
    rpts = camera_matrix @ (tx_camera_tag @ pts)[:3,:]
    rpts /= rpts[2,:]
    return rpts[:2,:]

def project_line_segment(camera_matrix, tx_camera_tag, x1, y1, x2, y2):
    pts = np.empty((4, 2))        
    pts[:,0] = [x1, y1, 0, 1]
    pts[:,1] = [x2, y2, 0, 1]
    rpts = camera_matrix @ (tx_camera_tag @ pts)[:3,:]
    rpts /= rpts[2,:]
    return rpts

def overlay_tag(overlayable, tag_corners, tag_id = None, thickness = 1):
    side_colors = [
        imgui.get_color_u32_rgba(1,0,0,1),
        imgui.get_color_u32_rgba(0,1,0,1),
        imgui.get_color_u32_rgba(0,0,1,1),
        imgui.get_color_u32_rgba(1,1,0,1)
    ]
    
    acenter = (np.sum(tag_corners, axis=1)/4).flatten()
    for i in range(4):
        color = side_colors[i]
        p0 = tag_corners[:,i].flatten()
        p1 = tag_corners[:,(i+1)%4].flatten()
        overlay_line(overlayable,
                     p0[0], p0[1], p1[0], p1[1],
                     color, thickness)

    if tag_id is not None:
        overlay_text(overlayable, acenter[0], acenter[1], imgui.get_color_u32_rgba(1,0,0,1), str(tag_id))


def overlay_line_list(overlayable, line_list, color, thickness):
    for i in range(int(line_list.shape[1]/2)):
        p = 2*i
        np = 2*i + 1
        overlay_line(overlayable,
                     line_list[0,p],
                     line_list[1,p],
                     line_list[0,np],
                     line_list[1,np],
                     color,
                     thickness)

def overlay_polyline(overlayable, polyline, color, thickness):
    for i in range(polyline.shape[1]):
        ni = (i + 1) % polyline.shape[1]
        overlay_line(overlayable,
                     polyline[0,i],
                     polyline[1,i],
                     polyline[0,ni],
                     polyline[1,ni],
                     color,
                     thickness)

def circle_fit_text(circle_fit):
    diameter = circle_fit.cr * 2
    return f"d= {diameter:#.4g} m"

def overlay_circle_fit_label(overlayable, projected_vertices, circle_fit, color):
    p0 = projected_vertices[:2,0]
    diameter = circle_fit.cr * 2
    overlay_text(overlayable,
                 p0[0] + 20,
                 p0[1],
                 color,
                 circle_fit_text(circle_fit))

def overlay_circle_fit(overlayable, projected_vertices, circle_fit, color = None, show_text = True):
    if color is None:
        color = imgui.get_color_u32_rgba(1,0,0,1)
    overlay_polyline(overlayable, projected_vertices, color, 2)
    if show_text:
        overlay_circle_fit_label(overlayable, projected_vertices, circle_fit, color)

def measurement_line_label_str(x1, y1, x2, y2):
    pt0 = np.array([x1, y1])
    pt1 = np.array([x2, y2])
    distance = np.linalg.norm(pt1 - pt0)
    return f"{distance:#.4g} m"

def overlay_measurement_line_label(overlayable, rpts, x1, y1, x2, y2, color):
    direction = (rpts[:2,1] - rpts[:2,0])
    direction /= np.linalg.norm(direction) + 1e-6
    direction_perp = np.array([-direction[1], direction[0]])

    if direction_perp[0] < 0:
        direction_perp *= -1

    line_ctr = (rpts[:2,0] + rpts[:2,1])/2
    text_center = line_ctr + direction_perp * 20

    overlay_text(overlayable,
                 text_center[0], text_center[1],
                 color,
                 measurement_line_label_str(x1, y1, x2, y2)
    )

class CircleFit:
    def __init__(self):
        self.data_points = []
        self.num_segments = 30
        self.cx = 0
        self.cy = 0
        self.cr = 0
        self.vertices = make_ngon(0,0,0,self.num_segments)
        self.cross_home = np.array([
            [-1, 1, 0, 0],
            [ 0, 0,-1, 1],
            [ 0, 0, 0, 0],
            [ 1, 1, 1, 1],
        ], dtype=np.float64)
        self.cross = self.cross_home.copy()

    def update_vertices(self):
        self.vertices = make_ngon(self.cx,self.cy,self.cr,self.num_segments)

        # cross if for visualizing where the center is
        self.cross = self.cross_home.copy()
        self.cross[:3,:] *= self.cr * 0.12
        self.cross += np.array([[self.cx, self.cy, 0, 0]]).T

    def force_points(self):
        if len(self.data_points) < 3:
            self.data_points.append((self.cx + self.cr, self.cy))
            self.data_points.append((self.cx - self.cr, self.cy))
            self.data_points.append((self.cx, self.cy + self.cr))
            self.data_points.append((self.cx, self.cy - self.cr))

    def add_point(self, x, y):
        self.data_points.append((x,y))
        if len(self.data_points) >= 3:
            self.cx, self.cy, self.cr = fit_circle(self.data_points)
            self.update_vertices()

    def clear(self):
        self.__init__()

class LineSegment:
    def __init__(self, x, y):
        self.start = (x, y)
        self.end = (x, y)

def load_drawings(map_dir, tag_line_segments, tag_circles):
    try:
        with open(get_path(map_dir, "drawings.json"), "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"circles": {}, "lines": {}}

    circles = integerize_keys(data["circles"])
    lines = integerize_keys(data["lines"])

    tag_line_segments.clear()
    tag_circles.clear()

    for tag_id, cs in circles.items():
        for c in cs:
            print("loading circle", c, "of tag", tag_id)
            cfit = CircleFit()
            cfit.cx, cfit.cy, cfit.cr = c[0], c[1], c[2]
            cfit.update_vertices()
            cfit.force_points()
            tag_circles[tag_id].append(cfit)

    for tag_id, ls in lines.items():
        for l in ls:
            lseg = LineSegment(0, 0)
            lseg.start = (l[0], l[1])
            lseg.end = (l[2], l[3])
            tag_line_segments[tag_id].append(lseg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=str, help="directory containing source data used to make the map")
    parser.add_argument("--map", "-m", type=str, help="directory containing map.json and viewpoints.json if different than source_dir", default="")
    parser.add_argument("--device", "-d", type=int, help="cv2 video capture device for live mode", default=-1)
    parser.add_argument("--realsense", "-rs", action="store_true", help="use pyrealsense api instead of cv2 for video streaming", default=False)
    args = parser.parse_args()

    video_streaming = args.device >= 0 or args.realsense

    source_dir = args.source_dir
    map_dir = args.map or args.source_dir

    map_data = load_map(map_dir)
    map_type = map_data["map_type"]
    tag_data = {}
    for tag_id, pose in map_data["tag_locations"].items():
        if map_type == "2d":
            tag_data[tag_id] = xyt_to_SE3(np.array([pose]).T)
        elif map_type == "2.5d":
            tag_data[tag_id] = xytz_to_SE3(np.array([pose]).T)
        elif map_type == "3d":
            tag_data[tag_id] = np.array(pose)
        else:
            raise RuntimeError("Unhandled map type", map_type)

    default_tag_side_length = map_data["tag_side_lengths"]["default"]
    tag_side_lengths = map_data["tag_side_lengths"]
    
    viewpoints_data = load_viewpoints(map_dir)
    images = load_images(source_dir)

    image_ids = sorted(images.keys())
    tag_ids = sorted(map_data["tag_locations"].keys())

    selected_image_id = image_ids[0]
    selected_tag_id = tag_ids[0]

    image_camera_matrix = load_camera_matrix(source_dir)

    rectified_view = RectifiedTagView(800, 60)
    tag_line_segments = defaultdict(list)
    tag_circles = defaultdict(list)

    load_drawings(map_dir, tag_line_segments, tag_circles)

    show_help = True

    circle_in_progress = False
    line_in_progress = False

    highlighted_line_segment = None
    highlighted_circle_fit = None
    in_progress_line_segment = None
    in_progress_circle = None

    LINE_TOOL = 0
    CIRCLE_TOOL = 1
    measure_tool = LINE_TOOL
    
    app = ImguiSdlWrapper('Measure Tool', 1280, 720)
    for image_id, image in images.items():
        app.add_image(image_id, image)

    app.add_image("rectified_view", np.zeros((rectified_view.view_size_px,
                                              rectified_view.view_size_px,
                                              3)))

    video = None
    aruco_dict = None
    aruco_params = None
    tracker = None
    tracker_initted = False


    if video_streaming:
        camera_width = images[selected_image_id].shape[1]
        camera_height = images[selected_image_id].shape[0]

    video_camera_matrix = image_camera_matrix
    if args.realsense:
        print("Using realsense")
        import pyrealsense2 as rs
        rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color,
                                camera_width, camera_height,
                                rs.format.bgr8,
                                30)
        selection = rs_pipeline.start(rs_config)
        intrinsics = selection.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        fx, fy, cx, cy, = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        rs_camera_matrix = np.array([
            [fx, 0, cx],
            [ 0, fy, cy],
            [ 0, 0, 1],
        ])

        video_camera_matrix = rs_camera_matrix
        camera_matrix_delta = np.linalg.norm(rs_camera_matrix - image_camera_matrix)
        if camera_matrix_delta > 1e-3:
            print("Warning: video stream is using a different camera matrix than the one used for the still frames")
        
    elif args.device >= 0:
        print("Using cv2 device", args.device)
        video = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    if video_streaming:
        app.add_image("video", np.zeros((camera_height,
                                         camera_width, 3)))

        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        aruco_params = cv2.aruco.DetectorParameters_create()
        tracker = InsideOutTracker(video_camera_matrix, map_data, max_regularizer = 1e3)

    HIGHLIGHT_COLOR = imgui.get_color_u32_rgba(1,0.5,0.2,1)
    IN_PROGRESS_COLOR = imgui.get_color_u32_rgba(1,0.2,0.5,1)
    FINALIZED_COLOR = imgui.get_color_u32_rgba(1,0,0,1)

    while app.running:
        app.main_loop_begin()

        if video_streaming:
            aruco_ids = []
            aruco_corners = []
            aruco_corners_flat = []

            have_image = False
            if args.device >= 0:
                have_image, image = video.read()
            elif args.realsense:
                try:
                    frames = rs_pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    image = np.array(color_frame.get_data(), np.uint8)
                    have_image = True
                except RuntimeError:
                    have_image = False

            if have_image:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images["video"] = rgb_image
                app.update_image("video", rgb_image)

                # update tracker
                aruco_corners, aruco_ids, aruco_rejected = cv2.aruco.detectMarkers(
                    image,
                    aruco_dict,
                    parameters=aruco_params)
                aruco_ids = aruco_ids if aruco_ids is not None else [] # aruco_ids is sometimes annoyingly None

                aruco_corners = [c[0] for c in aruco_corners]
                aruco_ids = [i[0] for i in aruco_ids]
                aruco_corners_flat = [np.array([c.flatten()]).T for c in aruco_corners]

                if not tracker_initted and tracker.error > 100:
                    tracker.update(aruco_ids, aruco_corners_flat, force_update = False)
                else:
                    tracker_initted = True

                if tracker_initted:
                    tracker.update(aruco_ids, aruco_corners_flat, force_update = True)

            viewpoints_data["video"] = tracker.tx_world_viewpoint

            imgui.begin("Inside Out Tracking")
            imgui.text(f"video success {have_image}")
            imgui.same_line()
            imgui.text(f"# detections {len(aruco_ids)}")
            imgui.text(f"avg error {tracker.error/(len(aruco_ids) * 4 + 0.001):#.4g}")
            imgui.text(f"converged {tracker.converged_guess}")
            imgui.text(f"regularizer {tracker.regularizer:#.4g}")
            imgui.same_line()
            imgui.text(f"max regularizer {tracker.max_regularizer:#.4g}")

            if imgui.button("large regularizer"):
                tracker.max_regularizer = 1e9
            imgui.same_line()
            if imgui.button("small regularizer"):
                tracker.max_regularizer = 1e3

            display_width = imgui.get_window_width() - 10
            tid, w, h = app.get_image("video")
            
            overlayable = draw_overlayable_image(tid, w, h, display_width)
            for tag_id, corners in zip(aruco_ids, aruco_corners):
                overlay_tag(overlayable, corners.T, tag_id, thickness=1)
            imgui.end()
        
        imgui.begin("Select")
        display_width = imgui.get_window_width() - 10  
        
        expanded_image_ids = image_ids if not video_streaming else ["video"] + image_ids
        for image_id in expanded_image_ids:
            if image_id == "video":
                camera_matrix = video_camera_matrix
            else:
                camera_matrix = image_camera_matrix

            tid, w, h = app.get_image(image_id)
            overlayable = draw_overlayable_image(tid, w, h, display_width)

            img_click = None
            img_hover = None
            if imgui.is_item_hovered():
                mx, my = imgui.get_mouse_pos()
                img_hover = overlay_inv_transform(overlayable, mx, my)
                if imgui.is_item_clicked():
                    img_click = img_hover

            tx_world_viewpoint = viewpoints_data[image_id]
            for tag_id in tag_ids:
                tx_world_tag = tag_data[tag_id]
                tx_viewpoint_tag = SE3_inv(tx_world_viewpoint) @ tx_world_tag

                tag_dir = tx_viewpoint_tag[:3,2].copy()
                to_tag = tx_viewpoint_tag[:3,3].copy()
                to_tag /= np.linalg.norm(to_tag)
                tag_dp = np.dot(tag_dir, to_tag)
                if tag_dp >= 0:
                    # camera is looking at the back of tag
                    continue

                corners_mat = get_corners_mat(tag_side_lengths.get(tag_id, default_tag_side_length))
                projected_corners = camera_matrix @ (tx_viewpoint_tag @ corners_mat)[:3,:]
                projected_corners[:2,:] /= projected_corners[2,:]
                projected_corners = projected_corners[:2,:]
                tag_thickness = 1

                if (img_click or img_hover) and quad_contains_pt(projected_corners, img_hover):
                    if img_hover:
                        tag_thickness = 3
                    if img_click:
                        selected_tag_id = tag_id
                        selected_image_id = image_id

                overlay_tag(overlayable, projected_corners, tag_id, thickness=tag_thickness)                        

                for i, line in enumerate(tag_line_segments[tag_id]):
                    pts = np.empty((4, 2))        
                    px, py = line.start
                    qx, qy = line.end
                    segment_color = imgui.get_color_u32_rgba(1,0,0,1)
                    if highlighted_line_segment == (tag_id, i):
                        segment_color = HIGHLIGHT_COLOR
                    rpts = project_line_segment(camera_matrix, tx_viewpoint_tag, px, py, qx, qy)
                    overlay_polyline(overlayable, rpts, segment_color, 2)

                for i, circle_fit in enumerate(tag_circles[tag_id]):
                    num_circle_points = len(circle_fit.data_points)
                    if num_circle_points < 3:
                        continue
                    circle_color = imgui.get_color_u32_rgba(1,0,0,1)
                    if highlighted_circle_fit == (tag_id, i):
                        circle_color = HIGHLIGHT_COLOR
                    projected_vertices = camera_matrix @ (tx_viewpoint_tag @ circle_fit.vertices)[:3,:]
                    projected_vertices /= projected_vertices[2,:]
                    projected_cross = camera_matrix @ (tx_viewpoint_tag @ circle_fit.cross)[:3,:]
                    projected_cross /= projected_cross[2:,:]

                    overlay_polyline(overlayable, projected_vertices, circle_color, 2)
                    overlay_line_list(overlayable, projected_cross, circle_color, 2)

                    
        imgui.end()

        imgui.begin("Measure")
        if imgui.button("save all drawings"):
            circle_data = defaultdict(list)
            line_data = defaultdict(list)
            
            for tag_id, line_segments in tag_line_segments.items():
                for l in line_segments:
                    line_data[tag_id].append((l.start[0], l.start[1], l.end[0], l.end[1]))

            for tag_id, circle_fits in tag_circles.items():
                for c in circle_fits:
                    print("saving circle",(c.cx, c.cy, c.cr), "of tag",tag_id)
                    circle_data[tag_id].append((c.cx, c.cy, c.cr))

            save_data = {"circles": circle_data, "lines": line_data}
            with open(get_path(map_dir, "drawings.json"), "w") as f:
                json.dump(save_data, f)

        
        display_width = imgui.get_window_width() - 10  
        imgui.text(f"image {selected_image_id}, tag {selected_tag_id}")

        if imgui.radio_button("line", measure_tool == LINE_TOOL):
            measure_tool = LINE_TOOL
        imgui.same_line()
        if imgui.radio_button("circle", measure_tool == CIRCLE_TOOL):
            measure_tool = CIRCLE_TOOL

        if selected_image_id == "video":
            camera_matrix = video_camera_matrix
        else:
            camera_matrix = image_camera_matrix

        tid, w, h = app.get_image(selected_image_id)
        tx_world_viewpoint = viewpoints_data[selected_image_id]
        tx_world_tag = tag_data[selected_tag_id]
        tx_viewpoint_tag = SE3_inv(tx_world_viewpoint) @ tx_world_tag
        tag_side_length = tag_side_lengths.get(tag_id, default_tag_side_length)
        corners_mat = get_corners_mat(tag_side_length)
        projected_corners = camera_matrix @ (tx_viewpoint_tag @ corners_mat)[:3,:]
        projected_corners /= projected_corners[2,:]

        _, rectified_view.cx = imgui.slider_int("cx", rectified_view.cx, 0, rectified_view.view_size_px)
        _, rectified_view.cy = imgui.slider_int("cy", rectified_view.cy, 0, rectified_view.view_size_px)
        _, rectified_view.tag_side_length_px = imgui.slider_int("tag_side_length_px", rectified_view.tag_side_length_px, 50, 200)

        _, show_help = imgui.checkbox("show help", show_help)

        homog = rectified_view.get_homog(projected_corners[:2,:].T)
        rectified_image = cv2.warpPerspective(images[selected_image_id], homog, (rectified_view.view_size_px, rectified_view.view_size_px))
        hcamera_matrix = homog @ camera_matrix

        app.update_image("rectified_view", rectified_image)
        tid, w, h = app.get_image("rectified_view")
        overlayable_rectified = draw_overlayable_image(tid, w, h, display_width)
        hovering = imgui.is_item_hovered()
        if hovering:
            overlay_circle(overlayable_rectified, rx, ry, 5, imgui.get_color_u32_rgba(1,0,0,1), 1)
        mx, my = imgui.get_mouse_pos()
        rx, ry = overlay_inv_transform(overlayable_rectified, mx, my)
        hx, hy = rectified_view.get_metric_coords(rx, ry, tag_side_length)

        next_highlighted_line_segment = None
        next_highlighted_circle_fit = None

        for otag_id in tag_ids:
            tx_world_otag = tag_data[otag_id]
            tx_viewpoint_otag = tx_viewpoint_tag @ SE3_inv(tx_world_tag) @ tx_world_otag

            # display line segments
            for i, line in enumerate(tag_line_segments[otag_id]):
                in_progress = False
                highlighted = False
                if highlighted_line_segment == (otag_id, i):
                    highlighted = True
                    segment_color = HIGHLIGHT_COLOR
                elif in_progress_line_segment == (otag_id, i):
                    in_progress = True
                    segment_color = IN_PROGRESS_COLOR
                else:
                    segment_color = FINALIZED_COLOR

                pts = np.empty((4, 2))        
                px, py = line.start
                qx, qy = line.end
                rpts = project_line_segment(hcamera_matrix, tx_viewpoint_otag, px, py, qx, qy)
                segment_highlight = line_near_pt(rpts[0,0],
                                                 rpts[1,0],
                                                 rpts[0,1],
                                                 rpts[1,1],
                                                 rx, ry) \
                                                 and hovering \
                                                 and not line_in_progress \
                                                 and not circle_in_progress \
                                                 and next_highlighted_line_segment is None
                if segment_highlight:
                    next_highlighted_line_segment = (otag_id, i)

                overlay_polyline(overlayable_rectified, rpts, segment_color, 2)

                if highlighted or in_progress:
                    label = measurement_line_label_str(px, py, qx, qy)
                    if show_help:
                        if in_progress:
                            imgui.set_tooltip(f"{label}\n\nleft click to finish\nright click to abort")
                        else:
                            imgui.set_tooltip(f"{label}\n\nright click to delete")
                    else:
                        imgui.set_tooltip(label)

            # display circles
            circle_fits = tag_circles[otag_id]
            for fit_idx, circle_fit in enumerate(circle_fits):
                in_progress = False
                highlighted = False

                circle_color = imgui.get_color_u32_rgba(1,0,0,1)
                if highlighted_circle_fit == (otag_id, fit_idx):
                    highlighted = True
                    circle_color = HIGHLIGHT_COLOR
                elif in_progress_circle == (otag_id, fit_idx):
                    in_progress = True
                    circle_color = IN_PROGRESS_COLOR
                else:
                    circle_color = FINALIZED_COLOR

                num_circle_points = len(circle_fit.data_points)
                circle_pts = project_points(hcamera_matrix, tx_viewpoint_otag, circle_fit.data_points)

                if circle_in_progress and otag_id == selected_tag_id and fit_idx + 1 == len(circle_fits):
                    for i in range(num_circle_points):
                        overlay_circle_filled(
                            overlayable_rectified,
                            circle_pts[0,i],
                            circle_pts[1,i],
                            3,
                            circle_color)

                if num_circle_points >= 3:
                    projected_vertices = hcamera_matrix @ (tx_viewpoint_otag @ circle_fit.vertices)[:3,:]
                    projected_vertices /= projected_vertices[2,:]
                    projected_cross = hcamera_matrix @ (tx_viewpoint_otag @ circle_fit.cross)[:3,:]
                    projected_cross /= projected_cross[2:,:]

                    # check if this circle should be highlighted
                    if hovering \
                       and next_highlighted_line_segment is None \
                       and next_highlighted_circle_fit is None \
                       and not circle_in_progress \
                       and not line_in_progress:
                        for i in range(circle_fit.num_segments):
                            ni = (i + 1) % circle_fit.num_segments
                            if line_near_pt(projected_vertices[0,i],
                                            projected_vertices[1,i],
                                            projected_vertices[0,ni],
                                            projected_vertices[1,ni],
                                            rx, ry):
                                next_highlighted_circle_fit = (otag_id, fit_idx)
                                break


                    overlay_polyline(overlayable_rectified, projected_vertices, circle_color, 2)
                    overlay_line_list(overlayable_rectified, projected_cross, circle_color, 1)                    

                if in_progress or highlighted:
                    circle_text = ""
                    if num_circle_points >= 3:
                        circle_text = circle_fit_text(circle_fit)
                    if show_help:
                        if in_progress:
                            if num_circle_points < 3:
                                help_text = "click at least 3 times to define a circle\nright click to abort"
                            else:
                                help_text = "click near to add a point\nclick far to finish\nright click to abort"
                        else:
                            help_text = "right click to delete"
                        if circle_text:
                            imgui.set_tooltip(f"{circle_text}\n\n{help_text}")
                        else:
                            imgui.set_tooltip(help_text)
                    elif circle_text:
                        imgui.set_tooltip(circle_text)

        if measure_tool == CIRCLE_TOOL:
            line_in_progress = False
            in_progress_line_segment = None

            circle_fits = tag_circles[selected_tag_id]            
            if hovering and imgui.is_mouse_clicked():
                if not circle_in_progress:
                    circle_fits.append(CircleFit())
                    circle_in_progress = True
                    in_progress_circle = (selected_tag_id, len(circle_fits)-1)
                if circle_in_progress:
                    circle_fit = circle_fits[-1]
                    if len(circle_fit.data_points) < 3:
                        circle_fits[-1].add_point(hx, hy)
                    else:
                        # need to distinguish between adding a point
                        # or completing the current fit
                        dx = (hx - circle_fit.cx)/tag_side_length * rectified_view.tag_side_length_px
                        dy = (hy - circle_fit.cy)/tag_side_length * rectified_view.tag_side_length_px
                        ideal_r = circle_fit.cr / tag_side_length * rectified_view.tag_side_length_px
                        actual_r = (dx**2 + dy**2)**0.5
                        if abs(ideal_r - actual_r) > 50:
                            circle_in_progress = False
                            in_progress_circle = None
                        else:
                            circle_fits[-1].add_point(hx, hy)

            elif hovering and imgui.is_mouse_clicked(1):
                if circle_fits and circle_in_progress:
                    del circle_fits[-1]
                circle_in_progress = False
                in_progress_circle = None

        if measure_tool == LINE_TOOL:
            circle_in_progress = False
            in_progress_circle = None

            line_segments = tag_line_segments[selected_tag_id]

            if hovering and imgui.is_mouse_clicked(1) and line_in_progress:
                line_in_progress = False
                in_progress_line_segment = None
                del line_segments[-1]

            if hovering and line_in_progress:
                line_segments[-1].end = (hx, hy)

            if hovering and imgui.is_mouse_clicked():
                if line_in_progress:
                    line_in_progress = False
                    in_progress_line_segment = None
                else:
                    line_segments.append(LineSegment(hx, hy))
                    line_in_progress = True
                    in_progress_line_segment = (selected_tag_id, len(line_segments)-1)

        if imgui.is_mouse_clicked(1) and highlighted_line_segment is not None:
            htag_id, hseg_idx = highlighted_line_segment
            del tag_line_segments[htag_id][hseg_idx]

        if imgui.is_mouse_clicked(1) and highlighted_circle_fit is not None:
            htag_id, hfit_idx = highlighted_circle_fit
            del tag_circles[htag_id][hfit_idx]

        highlighted_line_segment = next_highlighted_line_segment
        highlighted_circle_fit = next_highlighted_circle_fit

        imgui.end()

        app.main_loop_end()

    app.destroy()

if __name__ == "__main__":
    main()
