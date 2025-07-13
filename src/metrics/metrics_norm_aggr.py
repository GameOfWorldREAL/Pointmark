from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from src.metrics.MetricData import Metric, METRICS

def aggregate_metrics(metric_list: list[Metric], point_cloud: np.ndarray, mode: str = "min", n: int = 5, num_threads: int = 1):
    if cpu_count() < num_threads:
        num_threads = cpu_count()

    def compute_single(measurement):
        kd_tree = cKDTree(point_cloud)
        return aggregate(measurement, point_cloud, mode, n, KDTree=kd_tree)

    n = min(len(metric_list), num_threads)
    print("number of threads used:", n)

    aggregated_metrics = Parallel(n_jobs=n, backend="loky")(
        delayed(compute_single)(metric.normalized) for metric in metric_list
    )
    for i, metric in enumerate(metric_list):
        if (
            not metric.name == METRICS.EDGE_POINTS or
            not metric.name == METRICS.OOI
        ):
            metric.normalized = aggregated_metrics[i]

def aggregate(data, point_cloud: np.ndarray, mode: str = "min", knn: int = 5, KDTree: cKDTree = None) :
    if mode == "none":
        return data

    if KDTree is None:
        if point_cloud is not None:
            KDTree = cKDTree(point_cloud)
        else:
            raise Exception("no point cloud or KDTree provided")

    mode_func = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean
    }.get(mode)
    if mode_func is None:
        raise Exception("unknown mode: ", mode)

    _, idx = KDTree.query(point_cloud, k=knn + 1)
    neighbor_data = data[idx]
    return mode_func(neighbor_data, axis=1)

def normalize(metrics_list: list[Metric], ooi_mask: np.ndarray):
    for metric in metrics_list:
        if metric.name == METRICS.CAMERA_DISTANCE:
            _, var = compute_mean_var(metric.measurement[ooi_mask], mean=0)
            metric.normalized = Q(metric.measurement, mean=0, var=var, invert=True)
        elif metric.name == METRICS.OBSERVATION_COUNT:
            metric.normalized = Q(metric.measurement, 2, 5)
        elif metric.name == METRICS.TRIANGULATION_UNCERTAINTY:
            ideal = 82.5 * np.pi / 180  #45° - 120° Luhmann-Nahbereichsfotogrammetrie
            var = 37.5 * np.pi / 180
            normQ = Q(metric.measurement, ideal, var, invert=True)
            maxTU = Q(np.pi, ideal, var, invert=True)
            print("TU: ", maxTU)
            metric.normalized = N(normQ, 0, maxTU)
        elif metric.name == METRICS.PROJECTION_ERROR:
            mean, var = compute_mean_var(metric.measurement[ooi_mask])
            metric.normalized = L(metric.measurement, mean, var)
        elif metric.name == METRICS.REPROJECTION_ERROR:
            mean, var = compute_mean_var(metric.measurement[ooi_mask])
            metric.normalized = L(metric.measurement, mean, var)
        elif metric.name == METRICS.DENSITY:
            mean, var = compute_mean_var(metric.measurement[ooi_mask])
            metric.normalized = L(metric.measurement, mean, var)
        elif metric.name == METRICS.ANGLE_OF_INCIDENCE:
            var = 45 * np.pi / 180
            normQ = Q(metric.measurement, 0, var, invert=True)
            maxAOI = Q(np.pi/2, 0, var, invert=True)
            print("AOI: ", maxAOI)
            metric.normalized = N(normQ, 0, maxAOI)
        elif metric.name == METRICS.DISTANCE_TO_EDGE:
            _, var = compute_mean_var(metric.measurement[ooi_mask], 0)
            metric.normalized = Q(metric.measurement, mean=0, var=var)
        elif metric.name == METRICS.EDGE_POINTS:
            metric.normalized = metric.measurement
        elif metric.name == METRICS.COMBINED:
            mean, var = compute_mean_var(metric.measurement[ooi_mask])
            metric.normalized = L(metric.measurement, mean, var)
        elif metric.name == METRICS.BRIGHTNESS:
            metric.normalized = metric.measurement
        elif metric.name == METRICS.DARKNESS:
            metric.normalized = metric.measurement
        elif metric.name == METRICS.IMG_QUALITY:
            metric.normalized = 1.0 - N(metric.measurement, 0, 5.6568)
        elif metric.name == METRICS.PRECISION:
            _, var = compute_mean_var(metric.measurement[ooi_mask], mean=0)
            metric.normalized = Q(metric.measurement, mean=0, var=var, invert=True)
        elif metric.name == METRICS.OOI:
            metric.normalized = metric.measurement
        else:
            raise Exception("unknown metric name")

def compute_mean_var(data, mean = None) -> tuple[np.float64, np.float64]:
    if mean is None:
        mean = np.sum(data) * (1.0 / data.shape[0])
    var2 = np.sum((data - mean) ** 2) * (1.0 / data.shape[0])
    var = np.sqrt(var2)
    return mean, var

def L(v, mean, var, invert = False):  #numpy compatible, v can be array
    def exp(x):
        x = np.clip(x, -87, 87) #overflow protection for very small and big exp float32
        return np.exp(x)
    if var < 1e-12:
        return v

    l =  1.0 / (1 + exp(-2 * (v - mean) / var))
    if invert:
        l = 1.0 - l
    return l

def Q(x, mean, var, invert = False):  #numpy compatible, x can be array
    q = (1.0 / (1.0 + ((x - mean) / var) ** 2))
    if invert:
        q = 1.0 - q
    return q

def N(x, min, max):
    norm = (x - min) / (max - min)
    return np.clip(norm, 0.0, 1.0)