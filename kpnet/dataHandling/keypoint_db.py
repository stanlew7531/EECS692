import attr
import os
from typing import List
import yaml
import numpy as np
from kpnet.utils.imgproc import PixelCoord, pixel_in_bbox
from kpnet.utils.transformations import quaternion_matrix
from kpnet.dataHandling.keypoint_db_entry import KeypointDBEntry


@attr.s
class KeypointDBConfig:
    pdc_data_root = ''

    config_file_path= None

    keypoint_yaml_name = 'scene_bbox_keypoint.yaml'

    verbose = True


def camera2world_from_map(camera2world_map) -> np.ndarray:
    """
    Get the transformation matrix from the storage map
    See ${processed_log_path}/processed/images/pose_data.yaml for an example
    :param camera2world_map:
    :return: The (4, 4) transform matrix from camera 2 world
    """
    # The rotation part
    camera2world_quat = [1, 0, 0, 0]
    camera2world_quat[0] = camera2world_map['quaternion']['w']
    camera2world_quat[1] = camera2world_map['quaternion']['x']
    camera2world_quat[2] = camera2world_map['quaternion']['y']
    camera2world_quat[3] = camera2world_map['quaternion']['z']
    camera2world_quat = np.asarray(camera2world_quat)
    camera2world_matrix = quaternion_matrix(camera2world_quat)

    # The linear part
    camera2world_matrix[0, 3] = camera2world_map['translation']['x']
    camera2world_matrix[1, 3] = camera2world_map['translation']['y']
    camera2world_matrix[2, 3] = camera2world_map['translation']['z']
    return camera2world_matrix

class KeypointDB():
    def __init__(self, config:KeypointDBConfig):

        self._config=config

        # load the requested scene paths
        self._scene_path_list = []
        if config.config_file_path is not None:
            self._scene_path_list = self._get_scene_from_config(config)
        else:
            raise NotImplementedError

        #loop over the scenes and process them
        self._keypoint_entries = []
        self._num_keypoints = -1
        for scene_path in self._scene_path_list:
            if config.verbose:
                print("Keypoint DB Processing: ", scene_path)
            # actual processing section
            scene_keypoint_entries = self._build_scene_entry(scene_path, config.keypoint_yaml_name)
            for keypoint_item in scene_keypoint_entries:
                self._keypoint_entries.append(keypoint_item)

        print('Total number of images/keypoint entries is {0}'.format(len(self._keypoint_entries)))

    def get_entry_list(self) -> List[KeypointDBEntry]:
        return self._keypoint_entries

    @property
    def num_keypoints(self):
        assert self._num_keypoints > 0
        return self._num_keypoints

    # god I love python code with strongly typed input/output vals....
    def _get_scene_from_config(self, config: KeypointDBConfig) -> List[str]:
        assert os.path.exists(config.pdc_data_root)
        assert os.path.exists(config.config_file_path)

        #read in the config file
        scene_root_list = []
        config_data = None
        with open(config.config_file_path, 'r') as config_fh:
            config_data = config_fh.read().split('\n')
        
        # make sure we got data from the config file
        assert config_data is not None

        # loop through the config entries and add the paths to the return value
        for line in config_data:
            if(len(line) == 0):
                continue
            scene_root = os.path.join(config.pdc_data_root, line)
            if self._is_scene_valid(scene_root, config.keypoint_yaml_name):
                scene_root_list.append(scene_root)
            else:
                print("invalid scene entry: {0}".format(scene_root))

        return scene_root_list

    # a scene is valid if it meets the required file structure
    # that is, the base scene root path exists, and a keypoint YAML file exists
    # within the relevant /processed/ subfolder
    def _is_scene_valid(self, scene_root:str, keypoint_yaml_name: str) -> bool:
        if not os.path.exists(scene_root):
            return False

        scene_root_plus_processed = os.path.join(scene_root, 'processed')
        keypoint_yaml_file_path = os.path.join(scene_root_plus_processed, keypoint_yaml_name)

        if not os.path.exists(keypoint_yaml_file_path):
            return False
        
        return True
    
    def _build_scene_entry(self, scene_root:str, keypoint_yaml_name: str) -> List[KeypointDBEntry]:
        # get & load the YAML file
        scene_root_plus_processed = os.path.join(scene_root, 'processed')
        keypoint_yaml_file_path = os.path.join(scene_root_plus_processed, keypoint_yaml_name)
        assert os.path.exists(keypoint_yaml_file_path)

        keypoint_yaml_file_handle = open(keypoint_yaml_file_path, 'r')
        keypoint_yaml_data = yaml.load(keypoint_yaml_file_handle)
        keypoint_yaml_file_handle.close()
        
        # iterate over all images
        entry_list = []
        for image_key in keypoint_yaml_data.keys():
            image_map = keypoint_yaml_data[image_key]
            image_entry = self._get_image_entry(image_map, scene_root)
            if image_entry is not None and self._check_image_entry(image_entry):
                entry_list.append(image_entry)

        return entry_list

    def _get_image_entry(self, image_map, scene_root: str) -> KeypointDBEntry:
        entry = KeypointDBEntry()
        # The path for rgb image
        rgb_name = image_map['rgb_image_filename']
        rgb_path = os.path.join(scene_root, 'processed/images/' + rgb_name)
        assert os.path.exists(rgb_path)
        entry.rgb_image_path = rgb_path

        # The path for depth image
        depth_name = image_map['depth_image_filename']
        depth_path = os.path.join(scene_root, 'processed/images/' + depth_name)
        assert os.path.exists(depth_path)
        entry.depth_image_path = depth_path

        # The path for mask image
        mask_name = depth_name[0:6] + '_mask.png'
        mask_path = os.path.join(scene_root, 'processed/image_masks/' + mask_name)
        assert os.path.exists(mask_path)
        entry.binary_mask_path = mask_path

        # The camera pose in world
        camera2world_map = image_map['camera_to_world']
        entry.camera_in_world = camera2world_from_map(camera2world_map)

        # The bounding box
        top_left = PixelCoord()
        bottom_right = PixelCoord()
        top_left.x, top_left.y = image_map['bbox_top_left_xy'][0], image_map['bbox_top_left_xy'][1]
        bottom_right.x, bottom_right.y = image_map['bbox_bottom_right_xy'][0], image_map['bbox_bottom_right_xy'][1]
        entry.bbox_top_left = top_left
        entry.bbox_bottom_right = bottom_right

        # The size of keypoint
        keypoint_camera_frame_list = image_map['3d_keypoint_camera_frame']
        n_keypoint = len(keypoint_camera_frame_list)
        if self._num_keypoints < 0:
            self._num_keypoints = n_keypoint
        else:
            assert self._num_keypoints == n_keypoint

        # The keypoint in camera frame
        entry.keypoint_camera = np.zeros((3, n_keypoint))
        for i in range(n_keypoint):
            for j in range(3):
                entry.keypoint_camera[j, i] = keypoint_camera_frame_list[i][j]

        # The pixel coordinate and depth of keypoint
        keypoint_pixelxy_depth_list = image_map['keypoint_pixel_xy_depth']
        assert n_keypoint == len(keypoint_pixelxy_depth_list)
        entry.keypoint_pixelxy_depth = np.zeros((3, n_keypoint), dtype=np.int)
        for i in range(n_keypoint):
            for j in range(3):
                entry.keypoint_pixelxy_depth[j, i] = keypoint_pixelxy_depth_list[i][j]

        # Check the validity
        entry.keypoint_validity_weight = np.ones((3, n_keypoint))
        for i in range(n_keypoint):
            pixel = PixelCoord()
            pixel.x = entry.keypoint_pixelxy_depth[0, i]
            pixel.y = entry.keypoint_pixelxy_depth[1, i]
            depth_mm = entry.keypoint_pixelxy_depth[2, i]
            valid = True
            if depth_mm < 0:  # The depth cannot be negative
                valid = False

            # The pixel must be in bounding box
            if not pixel_in_bbox(pixel, entry.bbox_top_left, entry.bbox_bottom_right):
                valid = False

            # Invalid all the dimension
            if not valid:
                entry.keypoint_validity_weight[0, i] = 0
                entry.keypoint_validity_weight[1, i] = 0
                entry.keypoint_validity_weight[2, i] = 0
                entry.on_boundary = True

        # OK
        return entry

    def _check_image_entry(self, entry: KeypointDBEntry) -> bool:
        # Check the bounding box
        if entry.bbox_top_left.x is None or entry.bbox_top_left.y is None:
            return False

        if entry.bbox_bottom_right.x is None or entry.bbox_bottom_right.y is None:
            return False

        # OK
        return True