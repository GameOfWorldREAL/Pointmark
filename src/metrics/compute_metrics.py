import argparse
import time
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
import torch
from joblib import Parallel, delayed

import src.metrics.metrics as mt
from src.metrics.OOI.object_of_interest import compute_object_of_interest
from src.ReconstructionData.ReconstructionData import ReconstructionData
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
from src.metrics.metrics import compute_density
from src.pointmark.setup.selectFunctions import select_project_path
from src.metrics.MetricData import Metric, METRICS, MetricData
from src.metrics.metrics_norm_aggr import aggregate_metrics, normalize, compute_mean_var, Q, N, L
from src.pointmark.setup.setup import setup_UI
from src.utils.metrics_utils import sort_features_by_camera, undistort_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------global variables--------------------------
TASKS_PER_CORE = 120000
NUM_CORES = cpu_count()

NUM_METRICS = 12
NUM_BASE_THREADS = 7
NUM_PARALLEL_METRICS = 3
NUM_SEQUENTIAL_METRICS = NUM_BASE_THREADS - NUM_PARALLEL_METRICS

#-------------------helper function---------------------------
def compute_num_threads(num_points, tasks_per_core=None):
    global TASKS_PER_CORE
    if tasks_per_core is not None:
        TASKS_PER_CORE = tasks_per_core

    max_threads = NUM_CORES - NUM_SEQUENTIAL_METRICS
    num_metric_threads = max(1, min(num_points // TASKS_PER_CORE, max_threads // NUM_PARALLEL_METRICS))
    print("number of cores: ", NUM_CORES)
    return NUM_CORES, NUM_BASE_THREADS, TASKS_PER_CORE, num_metric_threads

#------------------wrapper functions--------------------------

def _compute_CD_OC_wrap(point_cloud, observation_offset ,observations, camera_pos):
    print("  compute camera distance and observation count metric")
    return mt.compute_CD_OC(point_cloud, observation_offset ,observations, camera_pos)

def _compute_TU_AIO_wrap(point_cloud, observation_offset ,observations, camera_pos, knn):
    print("  compute triangulation uncertainty and angle of incidence metric")
    return mt.compute_TU_AIO(point_cloud, observation_offset ,observations, camera_pos, knn=knn)

def _compute_PE_RPE_wrap(point_cloud, observations, observation_offset, camera_pos, intrinsics, camera_intrinsics_map, features):
    print("  compute projection and reprojection error metric")
    camera_features = sort_features_by_camera(observations, camera_pos.shape[0])
    undistorted_features = undistort_features(features, camera_features, intrinsics, camera_intrinsics_map)
    return mt.compute_PE_RPE(point_cloud, observations, observation_offset, camera_pos, intrinsics, camera_intrinsics_map, undistorted_features)

def _compute_ED_EP_wrap(point_cloud, density: np.ndarray, knn):
    print("  compute distance to edge and edge points metric")
    return mt.compute_ED_EP(point_cloud, density, knn=round(knn*1.5))

def _compute_BRT_DRK_wrap(point_cloud, observations, observation_offset, features, image_paths):
    print("  compute brightness, darkness and image quality metric")
    return mt.compute_BRT_DRK(point_cloud, observations, observation_offset, features, image_paths, device)

def _compute_IQ_wrap(point_cloud, observations, observation_offset, features, image_paths):
    print("  compute image quality metric")
    return mt.compute_IQ(point_cloud, observations, observation_offset, features, image_paths, device)

def _compute_PR_wrap(point_cloud, density):
    print("  compute precision metric")
    return mt.compute_PR(point_cloud, density)

#-------------------combine------------------------------
#TODO non weighted combine function
def combine_metrics(metric_list: list[Metric]):
    combined = np.zeros_like(metric_list[0].get_normalized())
    for metric in metric_list:
        if metric.name == METRICS.CAMERA_DISTANCE:
            mean, var = compute_mean_var(metric.measurement)
            norm = Q(metric.measurement, mean, var=var, invert=True)
            combined += norm
        elif metric.name == METRICS.DENSITY:
            m_min = np.min(metric.measurement)
            m_max = np.max(metric.measurement)
            norm = N(metric.measurement, m_min, m_max)
            combined += norm

        elif metric.name == METRICS.TRIANGULATION_UNCERTAINTY:
            mean = 45
            var = 37.5
            norm = L(metric.measurement, mean, var)
            combined += norm
        elif metric.name == METRICS.DISTANCE_TO_EDGE:
            combined += metric.normalized
        elif metric.name == METRICS.ANGLE_OF_INCIDENCE:
            combined += metric.normalized
        elif metric.name == METRICS.OBSERVATION_COUNT:
            combined += metric.normalized
        elif metric.name == METRICS.IMG_QUALITY:
            combined += metric.normalized

    #nomrlaize
    combined = combined / 7
    return combined

#--------------------functions----------------------------
def build_metrics(metric_data: MetricData, pointmark_paths: PointmarkPaths, aggregation_mode="min", ooi=True, save=False):
    if metric_data.has_save_data():
        metric_data.restore()

        if metric_data.get_aggregation_mode() != aggregation_mode:
            reaggregate(pointmark_paths, metric_data.project_paths, aggregation_mode)
    else:
        metric_data.set_aggregation_mode(aggregation_mode)
        metric_list, score = compute_metrics(pointmark_paths, metric_data.project_paths, aggregation_mode, compute_ooi=ooi)
        metric_data.set_metrics(metric_list)
        metric_data.score = score
        if save:
            metric_data.save()
    return metric_data

#------------------main function--------------------------
def compute_metrics(pointmark_paths: PointmarkPaths, project_paths: ProjectPaths, aggregation_mode="none", compute_ooi=False):
    reconst_data = ReconstructionData(pointmark_paths, project_paths, silent_print=True)
    point_cloud = reconst_data.get_point_cloud()
    observations = reconst_data.get_observations()
    observation_offset = reconst_data.get_observation_offset()
    camera_pos = reconst_data.get_camera_poses()
    intrinsics = reconst_data.get_intrinsics()
    camera_intrinsics_map = reconst_data.get_camera_intrinsic_map()
    features = reconst_data.get_features()
    image_paths = reconst_data.get_cam_img_path()

    print("number of points: ", point_cloud.shape[0])
    print("number of cameras: ", camera_pos.shape[0])
    _, num_threads, _ , num_metric_threads = compute_num_threads(point_cloud.shape[0])
    mt.set_thread_data(num_threads, num_metric_threads)

    print("-----------------------------------------")
    start = time.time()
    start_total = start

    k = min(max(point_cloud.shape[0] // 12000, 20), 100)
    #compute initial density
    density = compute_density(point_cloud, knn=k)
    if compute_ooi:
        print("compute object of interest")
        print("  project path: ", project_paths.project_root_path)
        print("  device: ", device.type)
        ooi = compute_object_of_interest(pointmark_paths, project_paths, density)
    else:
        ooi = np.ones(point_cloud.shape[0], dtype=np.float32)
    ooi_points = sum(ooi)

    if compute_ooi:
        end = time.time()
        print("done")
        print("number of points in OOI: ", int(ooi_points))
        print("compute OOI took", end - start, "seconds")
        print("-----------------------------------------")


    # refine k:
    k = min(max(ooi_points // 1000, 20), 100)
    #refine density

    print("compute metrics")
    print("number of processes: ", num_threads + num_metric_threads*NUM_PARALLEL_METRICS)
    print("device: ", device.type)
    print("  compute density metric")
    density = compute_density(point_cloud, knn=k)
    # scale distances by density median

    results = Parallel(n_jobs=NUM_BASE_THREADS)([
        delayed(_compute_CD_OC_wrap)(point_cloud, observation_offset ,observations, camera_pos),
        delayed(_compute_TU_AIO_wrap)(point_cloud, observation_offset, observations, camera_pos, round(k*2.5)),
        delayed(_compute_PE_RPE_wrap)(point_cloud, observations, observation_offset, camera_pos, intrinsics, camera_intrinsics_map, features),
        delayed(_compute_ED_EP_wrap)(point_cloud, density, k),
        delayed(_compute_BRT_DRK_wrap)(point_cloud, observations, observation_offset, features, image_paths),
        delayed(_compute_IQ_wrap)(point_cloud, observations, observation_offset, features, image_paths),
        delayed(_compute_PR_wrap)(point_cloud, density)
    ])
    metrics = [
        results[0][0], results[0][1],                   # CD,  OC
        results[1][0], results[1][1],                   # TU,  AIO
        results[2][0], results[2][1],                   # PE,  RPE
        results[3][0], results[3][1],                   # ED,  EP
        results[4][0], results[4][1],                   # BRT, DRK
        results[5],                                     # IQ
        results[6]                                      # PR
    ]

    med = np.median(density)
    density = density / med

    camera_distance =           metrics[0]/med
    observation_count =         metrics[1]
    triangulation_uncertainty = metrics[2]
    angle_of_incidence =        metrics[3]
    projection_error =          metrics[4]/med
    reprojection_error =        metrics[5]/med
    distance_to_edge =          metrics[6]/med
    edge_points =               metrics[7]
    brightness =                metrics[8]
    darkness =                  metrics[9]
    image_quality =             metrics[10]
    precision =                 metrics[11]/med

    end = time.time()
    print("done")
    print(f"computation took {end - start} seconds")
    print("-----------------------------------------")

    metric_list = [Metric(METRICS.CAMERA_DISTANCE, camera_distance),
                   Metric(METRICS.OBSERVATION_COUNT, observation_count),
                   Metric(METRICS.TRIANGULATION_UNCERTAINTY, triangulation_uncertainty),
                   Metric(METRICS.ANGLE_OF_INCIDENCE, angle_of_incidence),
                   Metric(METRICS.PROJECTION_ERROR, projection_error),
                   Metric(METRICS.REPROJECTION_ERROR, reprojection_error),
                   Metric(METRICS.DENSITY, density),
                   Metric(METRICS.DISTANCE_TO_EDGE, distance_to_edge),
                   Metric(METRICS.EDGE_POINTS, edge_points),
                   Metric(METRICS.BRIGHTNESS, brightness),
                   Metric(METRICS.DARKNESS, darkness),
                   Metric(METRICS.IMG_QUALITY, image_quality),
                   Metric(METRICS.PRECISION, precision),
                   Metric(METRICS.OOI, ooi)]

    print("normalize metrics")
    start = time.time()
    normalize(metric_list, ooi.astype(np.int32))
    end = time.time()
    print("done")
    print(f"normalization took {end - start} seconds")
    print("-----------------------------------------")

    print("combine metrics")
    start = time.time()
    combined = combine_metrics(metric_list)
    metric_list.append(Metric(METRICS.COMBINED, combined, combined))
    score = np.mean(combined)
    end = time.time()
    print("done")
    print(f"combination took {end - start} seconds")
    print("-----------------------------------------")

    print("aggregate metrics")
    start = time.time()
    aggregate_metrics(metric_list, point_cloud, aggregation_mode, num_threads=NUM_METRICS)
    end = time.time()
    print("done")
    print(f"aggregation took {end - start} seconds")
    print("-----------------------------------------")

    end_total = time.time()
    print(f"total metric computation took {end_total - start_total} seconds")
    print("-----------------------------------------")
    return metric_list, score

def reaggregate(pointmark_paths: PointmarkPaths, project_paths: ProjectPaths, mode = "none"):
    metric_data = MetricData(project_paths)
    if not metric_data.has_save_data():
        print("no metrics found, compute metrics first")
    reconst_data = ReconstructionData(pointmark_paths, project_paths)
    point_cloud = reconst_data.get_point_cloud()
    metric_data.restore()
    metric_list = metric_data.get_metrics()
    metric_data.set_aggregation_mode(mode)
    aggregate_metrics(metric_list, point_cloud, mode, num_threads=NUM_METRICS)
    return metric_list

def main():
    parser = argparse.ArgumentParser(description="CLI for metrics computation")
    parser.add_argument("-p", "--project_path",
                        help="the folder the project data will be stored",
                        type=Path)
    parser.add_argument("-y", "--yes",
                        action="store_true",
                        help="all questions will be answered with yes")
    parser.add_argument('--no-save', action='store_true', help='dont save metrics to file')
    args = parser.parse_args()

    setup_UI()

    skip = args.yes
    no_save = args.no_save
    pointmark_paths = PointmarkPaths(Path(__file__).resolve().parents[2])
    if args.project_path is None:
        project_path = select_project_path(pointmark_paths, skip).project_root_path
    else:
        project_path = args.project_path

    project_paths = ProjectPaths(project_path)
    metric_data = MetricData(project_paths)

    build_metrics(metric_data, pointmark_paths, save=not no_save)

if __name__ == "__main__":
    main()