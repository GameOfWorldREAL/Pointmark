import copy
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
from matplotlib import cm, pyplot as plt
from probreg import cpd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import HuberRegressor

import src.Validation.reference_to_h5 as reference_to_h5

from src.ReconstructionData.ReconstructionData import ReconstructionData
from src.general.python.filePaths import ProjectPaths, PointmarkPaths
from src.metrics.MetricData import Metric, METRICS, MetricData
from src.utils.metrics_utils import compute_PCA


def get_aggregated_spread_curve(x, y, x_bins=50, y_bins=50, agg="iqr", smoothing=0.05,
                                 only_median=False, add_spread=False, rm_spread= False, reduction="median"):
    import numpy as np
    import scipy.ndimage

    if only_median and add_spread:
        raise ValueError("Cannot combine only_median=True and add_spread=True")

    x_edges = np.linspace(np.min(x), np.max(x), x_bins + 1)
    y_edges = np.linspace(np.min(y), np.max(y), y_bins + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    curve = []

    for i in range(y_bins):
        y_mask = (y >= y_edges[i]) & (y < y_edges[i + 1])
        x_in_y_bin = x[y_mask]

        if len(x_in_y_bin) == 0:
            curve.append(np.nan)
            continue

        if only_median:
            val = np.median(x_in_y_bin)
            curve.append(val)
            continue

        bin_medians = []
        for j in range(x_bins):
            x_mask = (x_in_y_bin >= x_edges[j]) & (x_in_y_bin < x_edges[j + 1])
            x_bin = x_in_y_bin[x_mask]
            if len(x_bin) > 0:
                bin_medians.append(np.median(x_bin))

        if len(bin_medians) == 0:
            curve.append(np.nan)
            continue

        bin_medians = np.array(bin_medians)

        if agg == "std":
            spread = np.std(bin_medians)
        elif agg == "var":
            spread = np.var(bin_medians)
        elif agg == "mad":
            spread = np.median(np.abs(bin_medians - np.median(bin_medians)))
        elif agg == "iqr":
            spread = np.percentile(bin_medians, 75) - np.percentile(bin_medians, 25)
        elif agg == "qspread":
            spread = np.percentile(bin_medians, 99) - np.percentile(bin_medians, 1)
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

        if add_spread:
            val = np.median(x_in_y_bin) + spread
        elif rm_spread:
            val = np.median(x_in_y_bin) - spread
        else:
            val = spread

        curve.append(val)

    curve = np.array(curve)

    # Interpolieren über NaNs
    valid = ~np.isnan(curve)
    if np.sum(valid) > 1:
        curve = np.interp(np.arange(len(curve)), np.where(valid)[0], curve[valid])
    else:
        curve[:] = 0.0

    curve_smooth = scipy.ndimage.gaussian_filter1d(curve, sigma=smoothing * y_bins)
    return y_centers, curve_smooth
def estimate_average_point_distance(pcd: o3d.geometry.PointCloud) -> float:
    distances = pcd.compute_nearest_neighbor_distance()
    return np.mean(distances)

def build_4x4_matrix(rot, t, scale):
    T = np.eye(4)
    T[:3, :3] = scale * rot
    T[:3, 3] = t
    return T

def to_numpy_if_needed(arr):
    # cupy is only imported if available
    if 'cupy' in str(type(arr)):
        return arr.get()
    return arr

def downsample_random(pcd: np.ndarray, max_points: int = 20000) -> np.ndarray:
    if pcd.shape[0] > max_points:
        idx = np.random.choice(pcd.shape[0], max_points, replace=False)
        return pcd[idx]
    return pcd

def binned_average(x, y, bins=100, use_median=False):
    bin_means_x = []
    bin_means_y = []
    bin_edges = np.linspace(np.min(x), np.max(x), bins + 1)
    for i in range(bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if np.any(mask):
            bin_means_x.append(np.mean(x[mask]))
            val = np.median(y[mask]) if use_median else np.mean(y[mask])
            bin_means_y.append(val)
    return np.array(bin_means_x), np.array(bin_means_y)

def get_measure(metric):
    if metric.name == METRICS.CAMERA_DISTANCE:
        return metric.measurement
    elif metric.name == METRICS.OBSERVATION_COUNT:
        return metric.measurement
    elif metric.name == METRICS.TRIANGULATION_UNCERTAINTY:
        return metric.measurement
    elif metric.name == METRICS.ANGLE_OF_INCIDENCE:
        return metric.measurement
    elif metric.name == METRICS.PROJECTION_ERROR:
        return metric.measurement
    elif metric.name == METRICS.REPROJECTION_ERROR:
        return metric.measurement
    elif metric.name == METRICS.DENSITY:
        return metric.measurement
    elif metric.name == METRICS.DISTANCE_TO_EDGE:
        return metric.measurement
    elif metric.name == METRICS.EDGE_POINTS:
        return metric.measurement
    elif metric.name == METRICS.COMBINED:
        return metric.measurement
    elif metric.name == METRICS.BRIGHTNESS:
        return metric.measurement
    elif metric.name == METRICS.DARKNESS:
        return metric.measurement
    elif metric.name == METRICS.PRECISION:
        return metric.measurement
    elif metric.name == METRICS.IMG_QUALITY:
        return metric.measurement
    elif metric.name == METRICS.OOI:
        return metric.measurement
    else:
        raise Exception("unknown metric: ", metric.name)


def get_norm(metric):
    if metric.name == METRICS.CAMERA_DISTANCE:
        return metric.normalized
    elif metric.name == METRICS.OBSERVATION_COUNT:
        return metric.normalized
    elif metric.name == METRICS.TRIANGULATION_UNCERTAINTY:
        return metric.normalized
    elif metric.name == METRICS.ANGLE_OF_INCIDENCE:
        return metric.normalized
    elif metric.name == METRICS.PROJECTION_ERROR:
        return metric.normalized
    elif metric.name == METRICS.REPROJECTION_ERROR:
        return metric.normalized
    elif metric.name == METRICS.DENSITY:
        return metric.normalized
    elif metric.name == METRICS.DISTANCE_TO_EDGE:
        return metric.normalized
    elif metric.name == METRICS.EDGE_POINTS:
        return metric.normalized
    elif metric.name == METRICS.COMBINED:
        return metric.normalized
    elif metric.name == METRICS.BRIGHTNESS:
        return metric.normalized
    elif metric.name == METRICS.DARKNESS:
        return metric.normalized
    elif metric.name == METRICS.PRECISION:
        return metric.normalized
    elif metric.name == METRICS.IMG_QUALITY:
        return metric.normalized
    elif metric.name == METRICS.OOI:
        return metric.normalized
    else:
        raise Exception("unknown metric: ", metric.name)


import open3d as o3d

def global_registration_ransac(src_points: np.ndarray,
                               tgt_points: np.ndarray,
                               voxel_size: float = 0.05):
    # 1) In Open3D-PointClouds umwandeln
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_points)
    tgt.points = o3d.utility.Vector3dVector(tgt_points)

    # 2) Downsample + Normalen schätzen
    src_ds = src.voxel_down_sample(voxel_size)
    tgt_ds = tgt.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    src_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius_normal, 30))
    tgt_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius_normal, 30))

    # 3) FPFH-Features berechnen
    radius_feature = voxel_size * 5
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius_feature, 100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius_feature, 100))

    # 4) RANSAC-Globalregistrierung
    distance_thresh = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_ds, tgt_ds,
        src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=400000, confidence=0.99))
    return np.asarray(result.transformation)


def rotate_pointcloud(
    alpha: float,
    axis: str = 'z',
    pointcloud: np.ndarray = None,
    ref_pointcloud: np.ndarray = None
):
    _, _, eigenvectors = compute_PCA(ref_pointcloud)
    local_axes = eigenvectors.T  # columns: x, y, z

    theta = np.radians(alpha)

    if axis == 'x':
        axis_idx = 0
    elif axis == 'y':
        axis_idx = 1
    elif axis == 'z':
        axis_idx = 2
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    rot_axis = local_axes[:, axis_idx]
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    K = np.array([
        [0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    T = np.eye(4)
    T[:3, :3] = R

    rotated = pointcloud @ R.T

    return rotated, T


def setup_pointcloud(pointcloud: np.ndarray, rotate_axis: int = None):
    # 1. PCA
    mean, eigenvalues, eigenvectors = compute_PCA(pointcloud)

    # 2. Optional: Flip (180° Rotation um eine Achse)
    if rotate_axis is not None:
        if rotate_axis < 0 or rotate_axis > 2:
            raise ValueError("rotate_axis must be 0, 1, or 2.")

        # Flip die beiden Achsen, die nicht die rotate_axis sind
        flip_matrix = np.eye(3)
        for i in range(3):
            if i != rotate_axis:
                flip_matrix[i, i] = -1
        eigenvectors = flip_matrix @ eigenvectors

    # 3. Rotation: von Welt-Koordinaten in PCA-Koordinaten
    R = eigenvectors.T  # Basiswechselmatrix

    # 4. Translation: in den Ursprung
    pointcloud_centered = pointcloud - mean

    # 5. Punktwolke rotieren
    rotated = pointcloud_centered @ R.T

    # 6. Skalierung berechnen
    flat = rotated.ravel()
    p_low, p_high = np.percentile(flat, [10, 90])
    scale = (1.0 / (p_high - p_low))

    # 7. Skaliert in PCA-Raum
    rotated_norm = rotated * scale

    # 8. Affine Transformationsmatrix T aufbauen
    # Korrekte Reihenfolge: erst Translation, dann Rotation, dann Skalierung
    T = np.eye(4)
    T[:3, :3] = scale * R  # Rotation und Skalierung kombiniert
    T[:3, 3] = -scale * (R @ mean)  # Translation wird zuerst angewendet

    return rotated_norm, T


class ValidationData:
    def __init__(self, pointmark_paths: PointmarkPaths, project_paths: ProjectPaths, groundTruth_path: Path=None, distances: np.ndarray=None):
        print("init validation data for project: ", project_paths.project_root_path.name)
        self._pointmark_paths = pointmark_paths
        self._project_paths = project_paths
        self._project_name = project_paths.project_root_path.name

        if self._project_name == "bunker":
            self._project_name = "Vange Well No. 5"
        if self._project_name == "GrabB":
            self._project_name = "Sarcophage Romain"
        if self._project_name == "baum_stumpf":
            self._project_name = "Spruce tree stump"

        self._store_path = self._project_paths.data_path / "validation.h5"

        metric_data = MetricData(project_paths)
        if metric_data.has_save_data():
            metric_data.restore()
            self._metric_list = metric_data.get_metrics()
        else:
            raise Exception("no metric data found")

        if groundTruth_path is not None:
            self.build_ground_truth(groundTruth_path)
        else:
            self.groundTruth = None

        if distances is not None:
            self.distances = distances
        else:
            self.distances = None

        self.pointcloud = None
        self.mask = None

        self._T = np.eye(4)
        print("-----------------------------------------")

    def build_ground_truth(self, groundTruth_path: Path):
        if not groundTruth_path.exists():
            raise Exception("no ground truth data found in:", groundTruth_path)
        mesh = o3d.io.read_triangle_mesh(groundTruth_path)
        ooi_metric = next((m for m in self._metric_list if m.name == METRICS.OOI), None)
        num_points = ooi_metric.measurement.shape[0]
        pcd = mesh.sample_points_uniformly(number_of_points=5*num_points)
        self.groundTruth = np.asarray(pcd.points)

    def set_metrics(self, metrics: list[Metric]):
        self._metric_list = metrics

    def get_metrics(self) -> list[Metric]:
        return self._metric_list

    def get_distances(self):
        return self.distances

    def set_distances(self, distances: np.ndarray):
        self.distances = distances

    def save(self):
        if self.distances is None:
            raise Exception("distances is None")
        if self.groundTruth is None:
            raise Exception("groundTruth is None")

        print("saving data at: ", self._store_path)
        data = {"distances": self.distances, "groundTruth": self.groundTruth, "T": self._T, "mask": self.mask}
        with h5py.File(self._store_path, "w") as file:
            for name, data_array in data.items():
                file.create_dataset(name, data=data_array, compression="gzip")
        print("-----------------------------------------")

    def restore(self):
        print("restoring data from: ", self._store_path)
        if not self.has_save_data():
            print("no saved metric data found.")
            return
        with h5py.File(self._store_path, 'r') as file:
            self.groundTruth = file["groundTruth"][:]
            self.distances = file["distances"][:]
            self._T = file["T"][:]
            self.mask = file["mask"][:]
        print("-----------------------------------------")

    def remove_save_data(self):
        if self.has_save_data():
            self._store_path.unlink()
            print("removed saved data")
        else:
            print("no saved data found")

    def has_save_data(self):
        print("has save data: ", self._store_path)
        return self._store_path.exists()

    def compute_distances(self, ground_truth_path: Path = None):
        print("compute distances")
        if self.groundTruth is None and ground_truth_path is None:
            raise Exception("no ground truth data found")
        elif ground_truth_path is not None:
            self.groundTruth = reference_to_h5.restore(ground_truth_path)
            #self.build_ground_truth(ground_truth_path)

        reconst_data = ReconstructionData(self._pointmark_paths, self._project_paths)
        reconst_pointcloud = reconst_data.get_point_cloud().astype(np.float64)

        print("  reconstruction dat: ", reconst_pointcloud.shape)
        print("  ground truth dat: ", self.groundTruth.shape)

        # filter
        print("  filter")
        ooi = self._metric_list[METRICS.OOI - 1].normalized
        mask = np.bool(ooi)
        #pr = self._metric_list[METRICS.PRECISION - 1].normalized
        #mask = mask | (pr <= 0.15)

        reconst_filtered = reconst_pointcloud[mask]

        _ , T1 = setup_pointcloud(reconst_filtered, rotate_axis=2) #val0 = 1 #val1 = 1 #val 2 = 1 #val3 = 0 #val4 = None
        groundTruth, _ = setup_pointcloud(self.groundTruth)
        self.groundTruth = groundTruth
        #self.groundTruth = 2 * groundTruth.mean(axis=0) - groundTruth  # 1 2 3

        T2 = np.eye(4)
        T3 = np.eye(4)
        T4 = np.eye(4)
        #groundTruth = downsample_random(self.groundTruth, max_points=2*reconst_filtered.shape[0])
        #val 0
        _, T2 = rotate_pointcloud(alpha=180, axis='z', pointcloud=reconst_filtered,ref_pointcloud=self.groundTruth)
        _, T3 = rotate_pointcloud(alpha=15, axis='y', pointcloud=reconst_filtered,ref_pointcloud=self.groundTruth)
        #_, T4 = rotate_pointcloud(alpha=-90, axis='x', pointcloud=reconst_filtered, ref_pointcloud=self.groundTruth)

        s = 1
        T5 = np.diag([s, s, s, 1])

        #val 1
        #T2 = np.eye(4)
        #val 2
        #_, T2 = rotate_pointcloud(alpha=-90, axis='z', pointcloud=reconst_filtered,ref_pointcloud=reconst_filtered)
        #val 3
        #_, T2 = rotate_pointcloud(alpha=180, axis='y', pointcloud=reconst_filtered, ref_pointcloud=reconst_filtered)
        #val 4
        #_ , T2 = rotate_pointcloud(alpha=-95, axis='x', pointcloud=reconst_filtered, ref_pointcloud=reconst_filtered)

        T2 = T5 @ T4 @ T3 @ T2
        T = T2 @ T1
        N = reconst_filtered.shape[0]
        reconst_filtered_hom = np.hstack([reconst_filtered, np.ones((N, 1))])
        reconst_filtered_hom = reconst_filtered_hom @ T.T
        reconst_filtered = reconst_filtered_hom[:, :3]

        N = reconst_pointcloud.shape[0]
        reconst_pointcloud_hom = np.hstack([reconst_pointcloud, np.ones((N, 1))])
        reconst_pointcloud_hom = reconst_pointcloud_hom @ T.T
        reconst_pointcloud = reconst_pointcloud_hom[:, :3]

        self.draw_distances(reconst_filtered, self.groundTruth)

        # initial registration with a probabilistic method
        print("  initial registration")

        # ref_down = downsample_random(groundTruth, max_points=10000)
        # reconst_down = downsample_random(np.asarray(reconst_filtered), max_points=10000)
        #
        # init_T_ransac = global_registration_ransac(
        #     ref_down,
        #     reconst_down,
        #     voxel_size=0.02
        # )
        #
        # rot = init_T_ransac[:3, :3]
        # t = init_T_ransac[:3, 3]
        # scale = 1.0
        #
        # T = build_4x4_matrix(rot, t, scale)
        #
        # self._T = T

        ref_down = downsample_random(self.groundTruth, max_points=8000)
        reconst_down = downsample_random(np.asarray(reconst_filtered), max_points=8000)

        tf_param, _, _ = cpd.registration_cpd(source=ref_down, target=reconst_down, tf_type_name='rigid',
                                              update_scale=True, use_cuda=True)
        rot = to_numpy_if_needed(tf_param.rot)
        t = to_numpy_if_needed(tf_param.t)
        scale = to_numpy_if_needed(tf_param.scale)
        T3 = build_4x4_matrix(rot, t, scale)

        ref_o3d = o3d.geometry.PointCloud()
        ref_o3d.points = o3d.utility.Vector3dVector(self.groundTruth)

        s = 1.4
        T5 = np.diag([s, s, s, 1])

        T3 = T5 @ T3

        self._T = T3
        self.draw_distances(reconst_filtered, self.groundTruth)

        reconst_o3d = o3d.geometry.PointCloud()
        reconst_o3d.points = o3d.utility.Vector3dVector(reconst_filtered)

        # icp registration
        print("  icp registration")
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=900)
        init_transformation = T3

        avg_dist = estimate_average_point_distance(reconst_o3d)
        max_corr_dist = 10 * avg_dist

        icp_result = o3d.pipelines.registration.registration_icp(
            ref_o3d, reconst_o3d, max_corr_dist, init_transformation, estimation, criteria
        )
        T4 = icp_result.transformation
        self._T = T4

        self.draw_distances(reconst_filtered, self.groundTruth)



        ref_point_cloud = self.groundTruth.astype(np.float32)
        ones = np.ones((self.groundTruth.shape[0], 1))
        groundTruth_hom = np.hstack([self.groundTruth, ones])
        groundTruth_hom = (self._T @ groundTruth_hom.T).T
        self.groundTruth = groundTruth_hom[:, :3].astype(np.float32)

        self._T = np.eye(4)
        reconst_pointcloud = reconst_pointcloud.astype(np.float32)
        #self.draw_distances(reconst_pointcloud, self.groundTruth)

        # compute point distance

        def oriented_bbox_minmax(points):
            c = points.mean(0)
            X = points - c
            _, _, vh = np.linalg.svd(X, full_matrices=False)
            R = vh  # Rotation PCA → Welt
            X_rot = X @ R.T
            mi = X_rot.min(0)
            ma = X_rot.max(0)
            return mi @ R + c, ma @ R + c

        min_groundTruth, max_groundTruth = oriented_bbox_minmax(self.groundTruth)


        mask = np.all((reconst_pointcloud >= min_groundTruth) & (reconst_pointcloud <= max_groundTruth), axis=1)

        mask = np.ones(reconst_pointcloud.shape[0], dtype=bool)
        self.pointcloud = reconst_pointcloud[mask]
        self.mask = mask


        print("  final")
        self.draw_distances(self.pointcloud, self.groundTruth)

        print("  compute point distance")
        kdtree = cKDTree(ref_point_cloud)
        distances, _ = kdtree.query(self.pointcloud, k=1, workers=-1)
        self.distances = distances


        print("-----------------------------------------")

    def plot(self, no_green = False, no_red =False, graph = True, normed = False, alpha = 0.1, percent = 99.99, bins = 500, smoothing = 0.02 ,only_metric: METRICS = None, ooi = False, project_name = None):

        if project_name is None:
            project_name = self._project_name

        if self.distances is None:
            raise Exception("distances is None")

        for metric in self._metric_list:

            if only_metric is not None:
                if metric.name != only_metric:
                    continue

            x = self.distances
            if normed:
                y = get_norm(metric)[self.mask]
            else:
                y = get_measure(metric)[self.mask]

            if ooi:
                ooi_m = self._metric_list[METRICS.OOI - 1]
                if ooi_m.name != METRICS.OOI:
                    raise Exception("ooi is not OOI", ooi_m.name)
                ooi_m = ooi_m.measurement.astype(np.bool)
                print(np.sum(ooi_m), "/", len(ooi_m))
                x = x[ooi_m]
                y = y[ooi_m]

            point_alpha = alpha
            point_size = 10 if len(x) < 100_000 else 2

            # --- Korrelationen ---
            pearson_corr, _ = pearsonr(x, y)
            spearman_corr, _ = spearmanr(x, y)

            # --- Huber-Regression (rot): y ~ x ---
            x_r = x.reshape(-1, 1)
            huber = HuberRegressor()
            huber.fit(x_r, y)
            y_huber = huber.predict(x_r)

            # --- Log(x)-Regression (blau) ---
            log_x_corr = None
            log_line_x = None
            log_line_y = None
            if not np.any(x <= 0):
                x_log = np.log1p(x).reshape(-1, 1)
                log_x_corr, _ = pearsonr(x_log.ravel(), y)

                huber_log = HuberRegressor()
                huber_log.fit(x_log, y)

                x_sorted = np.linspace(np.min(x), np.max(x), 500)
                x_sorted_log = np.log1p(x_sorted).reshape(-1, 1)
                y_log_huber = huber_log.predict(x_sorted_log)
                log_line_x = x_sorted
                log_line_y = y_log_huber
            else:
                print(f"{metric.name.name}: x enthält 0 oder negative Werte – log(x)-Regression übersprungen.")

            # --- Inverse(x)-Regression (lila) ---
            inv_x_corr = None
            inv_line_x = None
            inv_line_y = None
            if not np.any(x <= 0):
                x_inv = (1 / (x + 1e-8)).reshape(-1, 1)
                inv_x_corr, _ = pearsonr(x_inv.ravel(), y)

                huber_inv = HuberRegressor()
                huber_inv.fit(x_inv, y)

                x_sorted = np.linspace(np.min(x), np.max(x), 500)
                x_sorted_inv = (1 / (x_sorted + 1e-8)).reshape(-1, 1)
                y_inv_huber = huber_inv.predict(x_sorted_inv)
                inv_line_x = x_sorted
                inv_line_y = y_inv_huber
            else:
                print(f"{metric.name.name}: x enthält 0 – 1/x-Regression übersprungen.")

            # --- Binned Trendlinie (orange) ---
            #x_bin, y_bin = binned_average(x, y, bins=1000, use_median=True)

            # --- Plot ---
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, alpha=point_alpha, s=point_size, color='black', label="Datenpunkte")

            y_smooth, v_smooth = get_aggregated_spread_curve(x, y, x_bins=bins, y_bins=bins,agg="qspread", smoothing=smoothing, only_median=True)

            if not no_red:
                plt.plot(v_smooth, y_smooth, color='red', linestyle='--', linewidth=1.5, label='Geglätteter x Median')

            y_smooth, v_smooth = get_aggregated_spread_curve(x, y, x_bins=bins, y_bins=bins, agg="qspread",
                                                             smoothing=smoothing)
            plt.plot(v_smooth, y_smooth, color='blue', linestyle='--', linewidth=1.5, label='Geglättete x-Verteilung')

            y_smooth, v_smooth = get_aggregated_spread_curve(x, y, x_bins=bins, y_bins=bins,agg="qspread", smoothing=smoothing, add_spread=True)
            if not no_green:
                plt.plot(v_smooth, y_smooth, color='green', linewidth=2, label='Geglättete Median + x-Verteilung  ')

            y_smooth, v_smooth = get_aggregated_spread_curve(x, y, x_bins=bins, y_bins=bins, agg="qspread",
                                                             smoothing=smoothing, rm_spread=True)
            #if not no_green:
            #    plt.plot(v_smooth, y_smooth, color='green', linewidth=2)



            #plt.plot(x_bin, y_bin, color='orange', linestyle='-', linewidth=1.5, label="Binned Trend")

            if graph:

                plt.plot(x, y_huber, color='red', linewidth=1.0, label="Huber: y ~ x")

                if log_line_x is not None:
                    plt.plot(log_line_x, log_line_y, color='blue', linestyle='--', linewidth=1.5,
                             label="Huber: y ~ log(x)")

                if inv_line_x is not None:
                    plt.plot(inv_line_x, inv_line_y, color='purple', linestyle='-.', linewidth=1.5,
                             label="Huber: y ~ 1/x")

            # Begrenzte Achsenskalierung
            if not normed:
                x_min, x_max = np.percentile(x, [0.001, 99.964])
                y_min, y_max = np.percentile(y, [0, percent])
            else:
                x_min, x_max = np.percentile(x, [0.001, 99.9])
                y_min = min(y)
                y_max = max(y)

            y_min = 0.15
            y_max = 0.65

            print(x_min, x_max, y_min, y_max)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xlabel("distance")
            if normed:
                if ooi:
                    plt.ylabel(f"{metric.name.name.lower()} (OOI) (normalisiert)")
                else:
                    plt.ylabel(f"{metric.name.name.lower()} (normalisiert)")
            else:
                if ooi:
                    plt.ylabel(f"{metric.name.name.lower()} (OOI)")
                else:
                    plt.ylabel(metric.name.name.lower())

            # Titel mit allen Korrelationen
            title = f"Pearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f}"
            if log_x_corr is not None:
                title += f" | log(x)-Pearson: {log_x_corr:.3f}"
            if inv_x_corr is not None:
                title += f" | 1/x-Pearson: {inv_x_corr:.3f}"

            plt.gcf().text(
                0.01, 0.01,  # Position relativ zur ganzen Figure (0 = ganz links/unten)
                f"{project_name}",  # Der Text
                ha='left',
                va='bottom',
                fontsize=10
            )
            #plt.title(title)

            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def draw_distances(self, source=None, target=None, dist = False):

        if self.groundTruth is None:
            raise Exception("groundTruth is None")

        reconst_data = ReconstructionData(self._pointmark_paths, self._project_paths)

        if source is None or target is None:
            source = self.pointcloud.astype(np.float64)
            target = self.groundTruth

        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(source)

        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target)

        source_temp = copy.deepcopy(pcd_source)
        target_temp = copy.deepcopy(pcd_target)

        if self.distances is not None:
            norm_d = self.distances / max(np.max(self.distances), 1e-8)
            colors = cm.plasma(norm_d)[:, :3]
            source_temp.colors = o3d.utility.Vector3dVector(colors)
            target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
        else:
            source_temp.paint_uniform_color([1.0, 0.55, 0.0])
            target_points = np.asarray(target_temp.points)
            z_vals = target_points[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            z_norm = (z_vals - z_min) / max((z_max - z_min), 1e-8)
            target_colors = cm.viridis(z_norm)[:, :3]
            target_temp.colors = o3d.utility.Vector3dVector(target_colors)

        if not dist:
            if self._T is None:
                o3d.visualization.draw_geometries([source_temp])
            else:
                target_temp.transform(self._T)
                o3d.visualization.draw_geometries([source_temp, target_temp])

        else:

            o3d.visualization.draw_geometries([source_temp])



