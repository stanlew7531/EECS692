import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
import attr
from kpnet.network.resnetBackbone import ResnetConfig, ResNetBackbone, resnet_spec
from kpnet.network.deconvHead import DeconvHead
from kpnet.network.resnetNoStage import ResnetNoStage
from kpnet.utils.imgproc import PixelCoord
import kpnet.utils.imgproc as imgproc
import kpnet.network.predict as predict
import kpnet.config.parameter as parameter
import numpy as np

@attr.s
class ImageProcOut(object):
    """
    Thin struct to hold the result of image processing
    """
    stacked_rgbd = np.ndarray(shape=[])
    bbox2patch = np.ndarray(shape=[])

    # Visualization only
    warped_rgb = np.ndarray(shape=[])
    warped_depth = np.ndarray(shape=[])

def processInferenceInputImages(cv_colorImg, cv_depthImg, bbox_corner, bbox_widthHeight):
    top_left, bottom_right = PixelCoord(), PixelCoord()
    top_left.x = bbox_corner[0]
    top_left.y = bbox_corner[1]
    bottom_right.x = top_left.x + bbox_widthHeight[0]
    bottom_right.y = top_left.y + bbox_widthHeight[1]

    # Crop the depth and RGB images as necessary based on inputted bounding boxes
    warped_rgb, bbox2patch = imgproc.get_bbox_cropped_image_raw(
            cv_colorImg, True,
            top_left, bottom_right,
            patch_width=parameter.default_patch_size_input, patch_height=parameter.default_patch_size_input,
            bbox_scale=parameter.bbox_scale)

    warped_depth, _ = imgproc.get_bbox_cropped_image_raw(
        cv_depthImg, False,
        top_left, bottom_right,
        patch_width=parameter.default_patch_size_input, patch_height=parameter.default_patch_size_input,
        bbox_scale=parameter.bbox_scale)

    # Perform normalization
    normalized_rgb = imgproc.rgb_image_normalize(warped_rgb, parameter.rgb_mean, [1.0, 1.0, 1.0])
    normalized_depth = imgproc.depth_image_normalize(
        warped_depth,
        parameter.depth_image_clip,
        parameter.depth_image_mean,
        parameter.depth_image_scale)

    # Construct the tensor
    channels, height, width = normalized_rgb.shape
    stacked_rgbd = np.zeros(shape=(channels + 1, height, width), dtype=np.float32)
    stacked_rgbd[0:3, :, :] = normalized_rgb
    stacked_rgbd[3, :, :] = normalized_depth

    # put into datastructure
    imgproc_out = ImageProcOut()
    imgproc_out.stacked_rgbd = stacked_rgbd
    imgproc_out.bbox2patch = bbox2patch
    imgproc_out.warped_rgb = warped_rgb
    imgproc_out.warped_depth = warped_depth

    return imgproc_out

def doInference(cv_colorImg, cv_depthImg, bbox_corner, bbox_widthHeight, network_weights_path):
    imgproc_result = processInferenceInputImages(cv_colorImg, cv_depthImg, bbox_corner, bbox_widthHeight)

    state_dict = torch.load(network_weights_path)
    n_keypoint = state_dict['head_net.features.9.weight'].shape[0] // 2

    net_config = ResnetConfig()
    net_config.num_keypoints = n_keypoint
    net_config.image_channels = 4
    net_config.depth_per_keypoint = 2
    net_config.num_layers = 34

    network = ResnetNoStage(net_config)
    network.load_state_dict(state_dict)
    network.cuda()
    network.eval()

    # Upload the image
    stacked_rgbd = torch.from_numpy(imgproc_result.stacked_rgbd)
    stacked_rgbd = torch.unsqueeze(stacked_rgbd, dim=0)
    stacked_rgbd = stacked_rgbd.cuda()
    
    # Do forward
    raw_pred = network(stacked_rgbd)
    assert raw_pred.shape[1] == 2 * n_keypoint
    prob_pred = raw_pred[:, 0:n_keypoint, :, :]
    depthmap_pred = raw_pred[:, n_keypoint:, :, :]
    heatmap = predict.heatmap_from_predict(prob_pred, n_keypoint)
    coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, n_keypoint)
    depth_pred = predict.depth_integration(heatmap, depthmap_pred)

    coord_x = coord_x.cpu().detach().numpy()
    coord_y = coord_y.cpu().detach().numpy()
    depth_pred = depth_pred.cpu().detach().numpy()

    keypointxy_depth_pred = np.zeros((3, n_keypoint))
    keypointxy_depth_pred[0, :] = coord_x[0, :, 0]
    keypointxy_depth_pred[1, :] = coord_y[0, :, 0]
    keypointxy_depth_pred[2, :] = depth_pred[0, :, 0]

    keypointxy_depth_realunit = np.zeros_like(keypointxy_depth_pred)
    keypointxy_depth_realunit[0, :] = (keypointxy_depth_pred[0, :] + 0.5) * parameter.default_patch_size_input
    keypointxy_depth_realunit[1, :] = (keypointxy_depth_pred[1, :] + 0.5) * parameter.default_patch_size_input
    keypointxy_depth_realunit[2, :] = (keypointxy_depth_pred[2, :] * parameter.depth_image_scale) + parameter.depth_image_mean

    keypoint_xy = keypointxy_depth_realunit[0:2, :]
    transform_homo = np.zeros((3, 3))
    transform_homo[0:2, :] = imgproc_result.bbox2patch
    transform_homo[2, 2] = 1
    patch2bbox = np.linalg.inv(transform_homo)

    keypoint_xy_homo = np.ones_like(keypointxy_depth_realunit)
    keypoint_xy_homo[0:2, :] = keypoint_xy
    keypoint_xy_depth_img = patch2bbox.dot(keypoint_xy_homo)
    keypoint_xy_depth_img[2, :] = keypointxy_depth_realunit[2, :]

    camera_vertex = np.zeros_like(keypoint_xy_depth_img)
    camera_vertex[2, :] = keypoint_xy_depth_img[2, :].astype(np.float) / 1000.0
    camera_vertex[0, :] = (keypoint_xy_depth_img[0, :] - parameter.principal_x) * camera_vertex[2, :] / parameter.focal_x
    camera_vertex[1, :] = (keypoint_xy_depth_img[1, :] - parameter.principal_y) * camera_vertex[2, :] / parameter.focal_y

    return camera_vertex