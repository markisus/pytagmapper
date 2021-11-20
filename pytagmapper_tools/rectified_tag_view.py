import cv2
import numpy as np

def get_rectified_tag_coords(tag_side_length_px, cx, cy):
    tag_coords = np.array([
        [-0.5, -0.5],
        [ 0.5, -0.5],
        [ 0.5,  0.5],
        [-0.5,  0.5],
    ], dtype=np.float64)
    tag_coords *= tag_side_length_px
    tag_coords += np.array([cx, cy], dtype=np.float64)
    return tag_coords

class RectifiedTagView:
    def __init__(self, view_size_px, tag_side_length_px):
        self.view_size_px = view_size_px
        self.tag_side_length_px = tag_side_length_px
        self.cx = self.view_size_px / 2
        self.cy = self.view_size_px / 2

    def get_homog(self, projected_corners):
        tag_coords = get_rectified_tag_coords(
            self.tag_side_length_px,
            self.cx, self.cy)        
        homog, ret = cv2.findHomography(projected_corners,
                                        tag_coords)
        return homog

    # metric coordinates in tag frame
    def get_metric_coords(self, view_x, view_y, tag_side_length):
        x = (view_x - self.cx) / self.tag_side_length_px * tag_side_length
        y = (view_y - self.cy) / self.tag_side_length_px * tag_side_length
        y *= -1
        return x, y

