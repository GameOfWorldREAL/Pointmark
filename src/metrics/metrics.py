import math

from multiprocessing import cpu_count, Pool
import cv2
import torch
from numba import njit
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import torchvision.transforms as transforms
from PIL import Image
import kornia.filters as k_filters

from src.metrics.OOI.object_of_interest import compute_radius
from src.utils.metrics_utils import inverse_K, compute_K, sort_feat_pointId_by_camera, compute_PCA

NUM_METRIC_THREADS = 1
NUM_CORES = cpu_count()

def set_thread_data(num_cores, num_metric_threads):
    global NUM_CORES, NUM_METRIC_THREADS
    NUM_CORES = num_cores
    NUM_METRIC_THREADS = num_metric_threads

#camera distance and observation count
# Ref: Javadnejad et al. (2021) – Dense point cloud quality factor
# Ref: Neumann et al. (2025) – Point cloud quality metrics
def compute_CD_OC(point_cloud: np.ndarray, observation_offset: np.ndarray, observations: np.ndarray, camera_poses: np.ndarray):

    counts = np.diff(observation_offset)    #gets the num of observations per point id
    point_pos_map = np.repeat(point_cloud, counts, axis=0) #multiplies the xyz cords with the num of observations
    obs_cam_ids = observations[:, 0]  #get the camera ids for observations, length is the same as observation_offset
    obs_cam_pos_map = camera_poses[obs_cam_ids, 9:12] #get camera pos for each cam id observed, now the same shape as point_pos_map
    diffs = obs_cam_pos_map - point_pos_map
    dists = np.linalg.norm(diffs, axis=1)

    sums = np.add.reduceat(dists, observation_offset[:-1])  #add all distances of an observation together
    cam_means = sums / counts
    return cam_means, counts

# Ref: Neumann et al. (2025) – Point cloud quality metrics
def compute_density(point_cloud: np.ndarray, knn: int = 20):
    KDTree = cKDTree(point_cloud)
    dists, _ = KDTree.query(point_cloud, k=knn + 1)
    return np.sum(dists, axis=1) / (knn + 1)


# Distance to edge
# Inspired by:
# Ref: Neumann et al. (2025) – Point cloud quality metrics
def compute_ED_EP(point_cloud: np.ndarray, density: np.ndarray, knn: int = 50):
    KDTree = cKDTree(point_cloud)

    _, idx = KDTree.query(point_cloud, k=knn + 1)
    neighbors = point_cloud[idx[:, 1:]].reshape(-1, 3)
    points_rep = np.repeat(point_cloud, knn, axis=0)
    dir_diff = (neighbors - points_rep).reshape(-1, knn, 3)

    norms = np.linalg.norm(dir_diff, axis=2, keepdims=True)
    dir_norms = dir_diff / np.where(norms == 0, 1e-8, norms)
    dir_mean = dir_norms.mean(axis=1)

    knn_spread = np.linalg.norm(dir_mean, axis=1)
    mask = (knn_spread > 0.45)

    edge_points = point_cloud[mask]
    edge_KDTree = cKDTree(edge_points)
    dists, _ = edge_KDTree.query(point_cloud, k=(knn // 10))
    dists = np.sum(dists, axis=1)  / (knn // 10)
    return dists/density, mask.astype(np.float32)

# Triangulation uncertainty
# Ref: Mauro et al. (2014) – View selection & planning (BMVC)
# Ref: Neumann et al. (2025) – Point cloud quality metrics

# Angle of incidence
# Ref: Javadnejad et al. (2021) – Dense point cloud quality factor
# Ref: Neumann et al. (2025) – Point cloud quality metrics

def compute_TU_AIO(point_cloud: np.ndarray, observation_offset: np.ndarray, observations: np.ndarray, camera_poses: np.ndarray, knn: int = 50):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd_normals = np.asarray(pcd.normals, dtype=np.float32)

    args = [
        (i, point_cloud, observation_offset, observations, camera_poses, pcd_normals)
        for i in range(len(point_cloud))
    ]

    c_size = max(1, math.ceil(len(point_cloud) / NUM_METRIC_THREADS))
    with Pool(processes=NUM_METRIC_THREADS) as pool:
        results = pool.starmap(_compute_single_tu_aoi, args, chunksize=c_size)

    tu_aio_np = np.array(results, dtype=np.float32)
    return tu_aio_np[:, 0], tu_aio_np[:, 1]

def _compute_single_tu_aoi(point_id, point_cloud: np.ndarray, observation_offset: np.ndarray, observations: np.ndarray,
                          camera_poses: np.ndarray, normals: np.ndarray):
    start = observation_offset[point_id]
    end = observation_offset[point_id + 1]
    point_observations_cam = observations[start:end, 0]
    normal_vector = normals[point_id]
    tu = _triangulation_uncertainty(point_cloud[point_id], point_observations_cam, camera_poses)
    aoi = _angle_of_incidence(point_cloud[point_id], normal_vector, point_observations_cam, camera_poses)
    return [tu, aoi]

@njit(cache=True)
def _angle_of_incidence(point_pos: np.ndarray, normal_vector: np.ndarray, point_observations_cam: np.ndarray, camera_poses: np.ndarray):
    angles = np.empty(len(point_observations_cam))
    for obsv_id in range(len(point_observations_cam)):
        cam_id = point_observations_cam[obsv_id]
        cam_pos = camera_poses[cam_id][9:]
        pc_v =  cam_pos - point_pos
        pc_n = pc_v / np.sqrt(np.dot(pc_v, pc_v))
        n_v = normal_vector / np.sqrt(np.dot(normal_vector, normal_vector))

        cross = np.cross(n_v, pc_n)
        cross_norm = np.sqrt(np.dot(cross, cross))
        dot = np.dot(n_v, pc_n)
        angle = np.abs(np.arctan2(cross_norm, dot))

        if angle > np.pi / 2:      #angle must be between 0 and pi/2
            angle = np.pi - angle
        angles[obsv_id] = angle

    return angles.sum() * (1.0 / angles.shape[0])

@njit(cache=True)
def _triangulation_uncertainty(point_pos: np.ndarray, point_observations: np.ndarray, camera_poses: np.ndarray):
    n = len(point_observations)
    angles = np.empty(n * (n - 1) // 2)
    i = 0

    for cam1 in range(n):
        for cam2 in range(cam1 + 1, n):
            cam_1_id = point_observations[cam1]
            cam_2_id = point_observations[cam2]
            cam_1_pos = camera_poses[cam_1_id][9:12]
            cam_2_pos = camera_poses[cam_2_id][9:12]

            v_1 = cam_1_pos - point_pos
            v_2 = cam_2_pos - point_pos
            n_1 = np.sqrt(np.sum(v_1 ** 2))
            n_2 = np.sqrt(np.sum(v_2 ** 2))
            cos = np.dot(v_1, v_2) / (n_1 * n_2)

            if cos > 1.0:
                cos = 1.0
            elif cos < -1.0:
                cos = -1.0

            angles[i] = np.arccos(cos)
            i += 1
    return angles.sum() * (1.0 / angles.shape[0])

# Projection error and Reprojection error
# Ref: Luhmann (2023) – Nahbereichsphotogrammetrie textbook
def compute_PE_RPE(point_cloud: np.ndarray, observations: np.ndarray, observation_offset: np.ndarray, camera_poses: np.ndarray,
               intrinsics: np.ndarray, camera_intrinsics_map: np.ndarray, features: np.ndarray):
    K_map = np.empty((camera_poses.shape[0], 3, 3), dtype=np.float32)
    for cam_id in range(camera_poses.shape[0]):
        cam_intrinsics = intrinsics[camera_intrinsics_map[cam_id]]
        K_map[cam_id] = compute_K(cam_intrinsics)

    args = [
        (i, point_cloud, observations, observation_offset, camera_poses, intrinsics, camera_intrinsics_map, features, K_map)
        for i in range(len(point_cloud))
    ]

    c_size = max(1, math.ceil(len(point_cloud) / NUM_METRIC_THREADS))
    with Pool(processes=NUM_METRIC_THREADS) as pool:
        results = pool.starmap(_compute_single_pe_rpe, args, chunksize=c_size)

    pe_rpe_np = np.array(results, dtype=np.float32)
    return pe_rpe_np[:, 0], pe_rpe_np[:, 1]

def _compute_single_pe_rpe(point_id, point_cloud: np.ndarray, observations: np.ndarray, observation_offset: np.ndarray, camera_poses: np.ndarray,
                           intrinsics: np.ndarray, camera_intrinsics_map: np.ndarray, features: np.ndarray, K_map: np.ndarray):
    start = observation_offset[point_id]
    end = observation_offset[point_id + 1]
    point_observations = observations[start:end]
    #compute projection error
    pe = _projection_error(point_cloud[point_id], point_observations, camera_poses, features, K_map)
    #compute reprojection error
    rpe = _reprojection_error(point_cloud[point_id], point_observations, camera_poses, intrinsics ,camera_intrinsics_map,features, K_map)
    return [pe, rpe]

@njit(cache=True)
def _projection_error(point_pos: np.ndarray, point_observations: np.ndarray, camera_poses: np.ndarray,
                      features: np.ndarray, K_map: np.ndarray):
    dists = np.empty(len(point_observations))
    for obsv_id in range(len(point_observations)):
        observation = point_observations[obsv_id]

        cam_id = observation[0]
        feature_id = observation[1]

        feature = features[feature_id]
        feat_x = feature[0]
        feat_y = feature[1]

        feat_p = np.array([feat_x, feat_y, 1.0], dtype=np.float32)

        R = camera_poses[cam_id][:9].reshape(3, 3)
        t = camera_poses[cam_id][9:]
        K = K_map[cam_id]

        K_inv = inverse_K(K)

        d_c = K_inv @ feat_p #vector from camera center to feature
        d_w = R @ d_c

        #compute point-line vector
        s = np.dot(point_pos - t, d_w) / np.dot(d_w, d_w)
        point_g = t + s * d_w

        distance_vector = point_g - point_pos
        dists[obsv_id] = np.sqrt(np.dot(distance_vector, distance_vector))  # norm njit compatible
    return dists.sum() * (1.0 / dists.shape[0])

@njit(cache=True)
def _reprojection_error(point_pos: np.ndarray, point_observations: np.ndarray, camera_poses: np.ndarray,
                        intrinsics: np.ndarray, camera_intrinsics_map: np.ndarray, features: np.ndarray, K_map: np.ndarray):
    errors_world = np.empty(len(point_observations))
    for obsv_id in range(len(point_observations)):
        observation = point_observations[obsv_id]

        cam_id = observation[0]
        feature_id = observation[1]

        feature = features[feature_id]
        feat_x = feature[0]
        feat_y = feature[1]

        feat_p = np.array([feat_x, feat_y], dtype=np.float32)

        R = camera_poses[cam_id][:9].reshape(3, 3)
        t = camera_poses[cam_id][9:]
        K = K_map[cam_id]

        #build inverse transformation matrix
        R_inv = R.T
        t_inv = -R_inv @ t
        T_inv = np.eye(4, dtype=np.float32)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv

        point_pos_h = np.empty(4, dtype=np.float32)
        point_pos_h[:3] = point_pos
        point_pos_h[3] = 1.0
        p_cam = T_inv @ point_pos_h
        p_cam = p_cam[:3]
        cam_dist = p_cam[2]

        p_img = K @ p_cam
        p_img_n = p_img / p_img[2]
        p_repr = p_img_n[:2]

        diff_v = p_repr - feat_p

        #compute pixel size in mm
        cam_intrinsics = intrinsics[camera_intrinsics_map[cam_id]]
        sensor_width = cam_intrinsics[4]
        sensor_height = cam_intrinsics[5]
        image_width = cam_intrinsics[6]
        image_height = cam_intrinsics[7]
        pixel_size_x = np.float32(sensor_width / image_width)
        pixel_size_y = np.float32(sensor_height / image_height)
        #norm in world units
        diff_v_mm = diff_v * np.array([pixel_size_x, pixel_size_y], dtype=np.float32)

        # scale error with radiation theorem
        fx_mm = K[0, 0] * pixel_size_x
        fy_mm = K[1, 1] * pixel_size_y
        err_world_v  = diff_v_mm * cam_dist / np.array([fx_mm, fy_mm], dtype=np.float32)
        err_world = np.sqrt(np.dot(err_world_v , err_world_v ))

        errors_world[obsv_id] = err_world  # norm njit compatible
    return errors_world.sum() * (1.0 / errors_world.shape[0])

# Brightness and Darkness
# Ref: Javadnejad et al. (2021) – Dense point cloud quality factor
def compute_BRT_DRK(point_cloud: np.ndarray, observations: np.ndarray, observation_offset: np.ndarray,
                   features: np.ndarray[np.int32, np.int32], image_paths: dict, device: torch.device = torch.device("cpu")):

    cam_feat_map = sort_feat_pointId_by_camera(point_cloud.shape[0], observation_offset, observations, len(image_paths))

    #cuda implementation with help from ChatGPT o3-mini
    if device.type == "cuda":
        features_gpu = torch.tensor(features, device=device)

        brt_rating_gpu = torch.zeros(point_cloud.shape[0], dtype=torch.float32, device=device)
        drk_rating_gpu = torch.zeros(point_cloud.shape[0], dtype=torch.float32, device=device)

        for str_cam_id, image_path in image_paths.items():
            cam_id = int(str_cam_id)
            image = cv2.imread(image_path)

            img_tensor = torch.tensor(image, dtype=torch.float32, device=device) / 255.0

            gray_gpu = 2 * (0.2126 * img_tensor[..., 2] +
                            0.7152 * img_tensor[..., 1] +
                            0.0722 * img_tensor[..., 0]) - 1

            brt_gpu = torch.clamp(gray_gpu, min=0)
            drk_gpu = torch.clamp(gray_gpu, max=0)

            feature_point_dict = cam_feat_map[cam_id]
            if feature_point_dict:
                feature_ids = torch.tensor(list(feature_point_dict.keys()), device=device)
                point_ids = torch.tensor(list(feature_point_dict.values()), device=device)

                coords = features_gpu[feature_ids]
                x_coords = coords[:, 0].round().long()
                y_coords = coords[:, 1].round().long()

                brt_rating_gpu.index_add_(0, point_ids, brt_gpu[y_coords, x_coords])
                drk_rating_gpu.index_add_(0, point_ids, drk_gpu[y_coords, x_coords])

            del img_tensor, gray_gpu, brt_gpu, drk_gpu

        brt_rating = brt_rating_gpu.cpu().numpy()
        drk_rating = drk_rating_gpu.cpu().numpy()

    else:
        brt_rating = np.zeros(point_cloud.shape[0], dtype=np.float32)
        drk_rating = np.zeros(point_cloud.shape[0], dtype=np.float32)
        for str_cam_id, image_path in image_paths.items():
            cam_id = int(str_cam_id)
            image = cv2.imread(image_path)
            B, G, R = cv2.split(image)
            brt, drk = process_image_cpu(B, G, R)

            #evaluate image
            for feature_id, point_id in cam_feat_map[cam_id].items():
                feature = features[feature_id]
                x = int(round(feature[0]))
                y = int(round(feature[1]))
                brt_rating[point_id] += brt[y, x]
                drk_rating[point_id] += drk[y, x]

    #normalize
    count = np.diff(observation_offset)
    brt_rating /= count
    drk_rating /= count

    return brt_rating, -drk_rating


@njit(parallel=True)
def process_image_cpu(B, G, R):
    gray = 2 * ((0.2126 * R + 0.7152 * G + 0.0722 * B) / 255) - 1
    brt = np.maximum(gray, 0)
    drk = np.minimum(gray, 0)
    return brt, drk

def compute_IQ(point_cloud: np.ndarray, observations: np.ndarray, observation_offset: np.ndarray,
               features: np.ndarray[np.int32, np.int32], image_paths: dict, device: torch.device = torch.device("cpu")):

    cam_feat_map = sort_feat_pointId_by_camera(point_cloud.shape[0], observation_offset, observations, len(image_paths))
    iq_rating = np.zeros(point_cloud.shape[0], dtype=np.float32)
    for str_cam_id, image_path in image_paths.items():
        cam_id = int(str_cam_id)
        # ---------------------------------------------------------
        # with help from ChatGPT 4o
        image = Image.open(image_path).convert("L")
        img = transforms.ToTensor()(image).unsqueeze(0).to(device)  # [1, 1, H, W]
        gradients = k_filters.spatial_gradient(img)  # [1, 1, 2, H, W]
        gx = gradients[:, :, 0, :, :]
        gy = gradients[:, :, 1, :, :]
        magnitude = torch.sqrt(gx ** 2 + gy ** 2).squeeze().cpu()
        # ---------------------------------------------------------
        for feature_id, point_id in cam_feat_map[cam_id].items():
            feature = features[feature_id]
            x = int(round(feature[0]))
            y = int(round(feature[1]))
            iq_rating[point_id] += magnitude[y, x].item()

    #normalize
    count = np.diff(observation_offset)
    iq_rating /= count

    return iq_rating

# Precision
# Ref: Luhmann (2023) – Nahbereichsphotogrammetrie textbook

def compute_PR(point_cloud: np.ndarray, density: np.ndarray):
    KDTree = cKDTree(point_cloud)
    radius = compute_radius(density, 4)

    neighbors = KDTree.query_ball_point(point_cloud, r=radius, workers=-1)
    args = [(point_cloud[i], point_cloud[neighbors[i]])
            for i in range(len(point_cloud))]

    c_size = max(1, math.ceil(len(point_cloud) / NUM_METRIC_THREADS))
    with Pool(processes=NUM_METRIC_THREADS) as pool:
        results = pool.starmap(_compute_single_pr, args, chunksize=c_size)

    dists = np.array(results, dtype=np.float32)

    #remove all np.inf values and replace with max distance
    finite_mask = np.isfinite(dists)
    finite_max = np.max(dists[finite_mask])
    dists[~finite_mask] = finite_max

    return dists

def _compute_single_pr(point: np.ndarray, neighbors: np.ndarray):
    unique_points = np.unique(neighbors, axis=0)
    if unique_points.shape[0] < 3:
        return np.inf
    mean, eigenvalues, eigenvectors = compute_PCA(neighbors)
    #distance to plane
    n = eigenvectors[2] # normed
    dist = np.abs(np.dot(point - mean, n))
    return dist
