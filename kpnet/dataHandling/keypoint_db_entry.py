import attr
import os
import numpy as np
from typing import List
from kpnet.utils.imgproc import PixelCoord, pixel_in_bbox

@attr.s
class KeypointDBEntry:
    # The path to rgb is must
    rgb_image_path = ''

    # The path to depth image
    depth_image_path = ''

    # The path to mask image
    binary_mask_path = ''

    # If length zero, indicates no depth
    @property
    def has_depth(self):
        return len(self.depth_image_path) > 0

    @property
    def has_mask(self):
        return len(self.binary_mask_path) > 0

    # The bounding box is tight
    bbox_top_left = PixelCoord()
    bbox_bottom_right = PixelCoord()

    # The information related to keypoint
    # All of these element should be in size of (3, n_keypoint)
    # The first element iterate over x, y, or z, the second element iterate over keypoints
    keypoint_camera = None  # The position of keypoint expressed in camera frame using meter as unit

    # (pixel_x, pixel_y, mm_depth) for each keypoint
    # Note that the pixel might be outside the image space
    keypoint_pixelxy_depth = None

    # Each element indicate the validity of the corresponded keypoint coordinate
    # 1 means valid, 0 means not valid
    keypoint_validity_weight = None
    on_boundary = False

    # The pose of the camera
    # Homogeneous transformation matrix
    camera_in_world = np.ndarray(shape=[4, 4])