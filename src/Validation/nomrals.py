import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
from tkinter import filedialog, Tk
from matplotlib import cm

from src.ReconstructionData.ReconstructionData import ReconstructionData
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
from src.metrics.MetricData import MetricData, METRICS


def mesh_to_pointcloud(ground_truth_path: Path, num_points: int = 1000000) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(ground_truth_path)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)

def get_reference_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference", type=Path)
    args, _ = parser.parse_known_args()

    if args.reference is not None:
        return args.reference
    else:
        Tk().withdraw()
        return Path(filedialog.askopenfilename(title="choose reference file:"))

def load_project_pointcloud_and_metrics(project_path: Path = None):
    if project_path is None:
        Tk().withdraw()
        project_path = Path(filedialog.askdirectory(title="choose project folder"))

    pointmark_paths = PointmarkPaths(project_path.parent)
    project_paths = ProjectPaths(project_path)

    reconst_data = ReconstructionData(pointmark_paths, project_paths)
    points = reconst_data.get_point_cloud()

    metric_data = MetricData(project_paths)
    if metric_data.has_save_data():
        metric_data.restore()
    else:
        raise RuntimeError("No saved metric data found in the project.")

    metrics = metric_data.get_metrics()

    return points, metrics

def remove_random_knn_region(points: np.ndarray, radius: float = 3) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # KDTree für Nachbarschaftssuche
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Zufälligen Punkt auswählen
    idx = np.random.randint(0, len(points))
    query_point = points[idx]

    # Finde alle Nachbarn im Radius
    [_, idxs, _] = kdtree.search_radius_vector_3d(query_point, radius)

    # Maske: Behalte alle Punkte, die NICHT in idxs sind
    mask = np.ones(len(points), dtype=bool)
    mask[idxs] = False

    return points[mask]

def visualize_pointcloud_with_normals(points: np.ndarray, sample_count: int =8000,
                                      radius: float = 0.5, max_nn: int = 30, normal_length: float = 3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)

    z = points[:, 2]
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-8)
    colors = cm.winter(z_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    idx = np.random.choice(len(points), size=min(sample_count, len(points)), replace=False)
    sampled_points = np.asarray(pcd.points)[idx]
    sampled_normals = np.asarray(pcd.normals)[idx]

    lines = []
    line_colors = []
    line_points = []

    for i, (p, n) in enumerate(zip(sampled_points, sampled_normals)):
        start = p
        end = p + normal_length * n
        line_points.append(start)
        line_points.append(end)
        lines.append([2 * i, 2 * i + 1])
        line_colors.append([1.0, 0.2, 0.2])  # kräftiges Lila

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    o3d.visualization.draw_geometries([pcd, line_set])



def main():
    #ref = get_reference_path()
    point_cloud = mesh_to_pointcloud(Path("F:\\Test\\Test_batch2\\projects\\bunker\\bunker_mod3.obj"))
    visualize_pointcloud_with_normals(point_cloud, normal_length=3)
    point_cloud, metrics = load_project_pointcloud_and_metrics(Path("F:\\Test\\Test_batch3\\projects\\GrabB"))
    #precision_metric = next((m for m in metrics if m.name == METRICS.PRECISION), None)

    mask = precision_metric.normalized < 0.4
    point_cloud = point_cloud[mask]

    print(f"point cloud shape: {point_cloud.shape}")
    visualize_pointcloud_with_normals(point_cloud, normal_length=0.02)

if __name__ == "__main__":
    main()