# WORK IN PROGRESS
from hack_sys_path import *

from imgui_sdl_wrapper import ImguiSdlWrapper
from collections import namedtuple
import imgui
import cv2
from gl_util import GlRgbTexture
import os
import argparse
from overlayable import *
import numpy as np

def overlay_aruco_corners(overlayable, aruco_ids, aruco_corners):
    aruco_side_colors = [
        imgui.get_color_u32_rgba(1,0,0,1),
        imgui.get_color_u32_rgba(0,1,0,1),
        imgui.get_color_u32_rgba(0,0,1,1),
        imgui.get_color_u32_rgba(1,1,0,1)
    ]

    for aruco_id, acorners in zip(aruco_ids, aruco_corners):
        acenter = (acorners[0] + acorners[1] + acorners[2] + acorners[3])/4
        for i in range(4):
            color = aruco_side_colors[i]
            p0 = acorners[i]
            p1 = acorners[(i+1)%4]
            overlay_line(overlayable,
                         p0[0], p0[1], p1[0], p1[1],
                         color, 1)
        overlay_text(overlayable, acenter[0], acenter[1], imgui.get_color_u32_rgba(1,0,0,1), str(aruco_id))

def captures_gui(ctx):
    if not imgui.begin("Captures"):
        imgui.end()
        return

    camera_width = ctx.image.shape[1]
    camera_height = ctx.image.shape[0]
    display_width = imgui.get_window_width() - 50
    scale = display_width/camera_width
    display_height = scale * camera_height

    ctx.delete_idx = None
    ctx.capture_idx_offset = 0

    _, ctx.save_dir = imgui.input_text("Save Dir", ctx.save_dir, 50)
    if imgui.button("Capture"):
        texture = GlRgbTexture(camera_width, camera_height)
        texture.update(cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB))
        ctx.captures.append(Capture(ctx.image, texture, ctx.aruco_ids, ctx.aruco_corners))
    imgui.same_line()
    ctx.should_save = imgui.button("Save")
    imgui.same_line()
    ctx.should_load = imgui.button("Load")
    for capture_idx in reversed(range(len(ctx.captures))):
        imgui.push_id(str(capture_idx))
        capture = ctx.captures[capture_idx]
        imgui.text(f"image_{capture_idx + ctx.capture_idx_offset}")
        imgui.same_line()
        if imgui.button("Delete"):
            ctx.delete_idx = capture_idx
        image = draw_overlayable_image(capture.texture.texture_id,
                                       capture.texture.width,
                                       capture.texture.height,
                                       display_width)
        overlay_aruco_corners(image, capture.aruco_ids, capture.aruco_corners)

        imgui.pop_id()
    imgui.end()

def camera_feed_gui(ctx):
    if not imgui.begin("Image Capture"):
        imgui.end()
        return
    display_width = imgui.get_window_width() - 50    
    image = draw_overlayable_image(ctx.camera_feed_texture.texture_id,
                                   ctx.camera_feed_texture.width,
                                   ctx.camera_feed_texture.height,
                                   display_width)
    overlay_aruco_corners(image, ctx.aruco_ids, ctx.aruco_corners)
    imgui.end()

class AppContext:
    pass

Capture = namedtuple("Capture", ["image", "texture", "aruco_ids", "aruco_corners"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=int, help="cv2 video capture device", default=-1)
    parser.add_argument("--realsense", "-rs", action="store_true", help="use pyrealsense api instead of cv2 for video streaming", default=False)
    
    args = parser.parse_args()
    camera_width = 1280
    camera_height = 720
    camera_matrix = None # populated from camera intrinsics if using realsense
    rs_pipeline = None # populated with realsense object if using realsense
    camera = None # populated with VideoCapture object if using opencv


    if args.device >= 0:
        device = args.device
        print("Using device", device)
        camera = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    elif args.realsense:
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
        camera_matrix = np.array([
            [fx, 0, cx],
            [ 0, fy, cy],
            [ 0, 0, 1],
        ])
        
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters_create()

    ctx = AppContext()
    ctx.captures = []
    ctx.save_dir = "data01"
    ctx.should_save = False
    ctx.should_load = False
    ctx.delete_idx = None
    
    app = ImguiSdlWrapper('Image Capture App', camera_width, camera_height)
    ctx.camera_feed_texture = GlRgbTexture(camera_width, camera_height)

    empty_image = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
    ctx.image = empty_image

    while app.running:
        app.main_loop_begin()

        if args.device >= 0:
            ret, ctx.image = camera.read()
        elif args.realsense:
            try:
                frames = rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                ctx.image = np.array(color_frame.get_data(), np.uint8)
                ret = True
            except RuntimeError:
                ret = False

        if ret:
            aruco_corners, aruco_ids, aruco_rejected = cv2.aruco.detectMarkers(
                ctx.image,
                aruco_dict,
                parameters=aruco_params)
            aruco_ids = aruco_ids if aruco_ids is not None else [] # aruco_ids is sometimes annoyingly None
        else:
            ctx.image = empty_image
            aruco_ids = []
            aruco_corners = []

        ctx.aruco_corners = [c[0] for c in aruco_corners]
        ctx.aruco_ids = [i[0] for i in aruco_ids]

        ctx.camera_feed_texture.update(cv2.cvtColor(ctx.image, cv2.COLOR_BGR2RGB))
        camera_feed_gui(ctx)
        captures_gui(ctx)
        app.main_loop_end()

        if ctx.delete_idx is not None:
            del ctx.captures[ctx.delete_idx]
        if ctx.should_save:
            if not os.path.exists(ctx.save_dir):
                os.mkdir(ctx.save_dir)
            for capture_idx, capture in enumerate(ctx.captures):
                cv2.imwrite(os.path.join(ctx.save_dir,f"image_{capture_idx + ctx.capture_idx_offset}.png"), capture.image)
            if camera_matrix is not None:
                with open(os.path.join(ctx.save_dir, "camera_matrix.txt"), "w") as f:
                    for row in camera_matrix:
                        f.write(" ".join(str(d) for d in row.tolist()) + "\n")

    del ctx
    app.destroy()

    if camera is not None:
        camera.release()

    if rs_pipeline is not None:
        rs_pipeline.stop()

if __name__ == "__main__":
    main()
