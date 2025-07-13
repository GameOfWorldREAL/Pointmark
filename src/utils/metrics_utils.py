import cv2
import numpy as np
from numba import njit


@njit(cache=True)
def compute_K(intrinsics: np.ndarray) -> np.ndarray:
    focal_length = intrinsics[0]
    sensor_width = intrinsics[4]
    sensor_height = intrinsics[5]
    image_width = intrinsics[6]
    image_height = intrinsics[7]
    ppx = intrinsics[8]
    ppy = intrinsics[9]

    # mm/px
    pixel_size_x =  sensor_width / image_width
    pixel_size_y =  sensor_height / image_height

    fx = focal_length / pixel_size_x
    fy =  focal_length / pixel_size_y

    cx = (image_width / np.float32(2.0)) + ppx
    cy = (image_height / np.float32(2.0)) + ppy

    # intrinsic matrix K
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return K

@njit(cache=True)
def inverse_K(K: np.ndarray) -> np.ndarray:
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    K_inv = np.array([1.0 / fx, 0.0, -cx / fx, 0.0, 1.0 / fy, -cy / fy, 0.0, 0.0, 1.0], dtype=np.float32).reshape(3, 3)
    return K_inv

#returns the undistorted feature set
def undistort_features(features: np.ndarray, camera_feature: list[list[int]], intrinsics: np.ndarray, camera_intrinsics_map: np.ndarray) -> np.ndarray:
    undistorted_features = np.empty_like(features)
    for cam_id, feature_id_list in enumerate(camera_feature):
        cam_intrinsics = intrinsics[camera_intrinsics_map[cam_id]]
        k1 = cam_intrinsics[1]
        k2 = cam_intrinsics[2]
        k3 = cam_intrinsics[3]
        k_dist = np.array([k1, k2, 0, 0, k3], dtype=np.float32)
        K = compute_K(cam_intrinsics)
        features_dist = features[feature_id_list]
        features_dist = features_dist.reshape(-1, 1, 2).astype(np.float32)

        undistort = cv2.undistortPoints(features_dist, K, k_dist, P=K)
        undistort = undistort.reshape(-1, 2)
        undistorted_features[feature_id_list] = undistort

    return undistorted_features

def sort_features_by_camera(observations, num_cameras):
    camera_buckets = [[] for _ in range(num_cameras)]  #build buckets
    for cam_id in range(num_cameras):
        mask = observations[:, 0] == cam_id  #mark observations of cam_id
        camera_buckets[cam_id] = observations[mask, 1].tolist() #store a list of feature ids in the bucket
    return camera_buckets

def sort_feat_pointId_by_camera(num_point_ids: int, observation_offset: np.ndarray, observations: np.ndarray, num_cam: int):
    camera_buckets = [{} for _ in range(num_cam)]  #build buckets
    point_ids = np.arange(0, num_point_ids)
    counts = np.diff(observation_offset)
    rep_ids = np.repeat(point_ids, counts)

    for cam_id in range(num_cam):
        mask = observations[:, 0] == cam_id  #mark observations of cam_id
        feature_ids = observations[mask, 1]
        point_ids = rep_ids[mask]
        camera_buckets[cam_id] = dict(zip(feature_ids, point_ids))
    return camera_buckets

def compute_PCA(point_cloud: np.ndarray):
    if point_cloud.shape[0] < 3:
        return np.zeros(3), np.zeros(3), np.zeros((3, 3))
    mean = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - mean
    cov = np.cov(centered_point_cloud.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].T
    return mean, eigenvalues, eigenvectors