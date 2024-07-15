""" Operations for point cloud processing """

import cv2
import numpy as np


def points2image(points: np.ndarray, intrinsics: np.ndarray, cam_resolution: tuple = None, extrinsics: np.ndarray = None) -> tuple:
    assert points.shape[1] == 3, "Points must be Nx3"
    
    if extrinsics is not None:
        points = transform_points(point_cloud=points, transformation=extrinsics)  # Transform points to camera frame before projecting

    points = points[points[:, 2] > 0.0]  # Filter out points behind camera
    
    uvs = np.dot(intrinsics, points.T).T
    uvs[:, :2] /= uvs[:, 2, np.newaxis]
    uv = np.rint(uvs[:, :2]).astype(int)
    
    if cam_resolution is None:
        cam_resolution = (int(intrinsics[0, 2] * 2), int(intrinsics[1, 2] * 2))
    
    valid_indices = (uv[:, 0] >= 0) & (uv[:, 0] < cam_resolution[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam_resolution[1])
    uv = uv[valid_indices]
    
    return uv


def transform_points(point_cloud: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    if point_cloud.shape[1] == 3:
        point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    
    transformed_points = np.dot(transformation, point_cloud.T).T
    
    return transformed_points[:, :3]


def get_lidar_projection_mask(point_cloud, cam_resolution, intrinsics, extrinsics=None, kernel_size=(10, 10)):
    point_cloud_uv = points2image(intrinsics=intrinsics, points=point_cloud[:, :3], extrinsics=extrinsics, cam_resolution=cam_resolution)
    
    mask = np.zeros((cam_resolution[1], cam_resolution[0]), dtype=np.uint8)
    mask[point_cloud_uv[:, 1], point_cloud_uv[:, 0]] = 1
    
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask
