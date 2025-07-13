from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

from src.ReconstructionData.ReconstructionData import ReconstructionData
from src.general.python.filePaths import ProjectPaths, PointmarkPaths
from src.metrics.MetricData import MetricData, METRICS
from src.metrics.compute_metrics import build_metrics

#=========================
#chatGPT 4.o generated gui
#=========================

pointmark_paths = PointmarkPaths(Path(__file__).resolve().parents[2])

class Project:
    def __init__(self, name, path):
        self.name = name
        self.path = Path(path)
        self.project_paths = ProjectPaths(Path(path))

        reconst_data = ReconstructionData(pointmark_paths, self.project_paths)
        self.points = reconst_data.get_point_cloud()
        self.camera_poses = reconst_data.get_camera_poses()

        if not self.project_paths.data_path.exists():
            raise Exception("project path seems to be invalid.")
        self.metric_data = MetricData(self.project_paths)
        build_metrics(self.metric_data, pointmark_paths, aggregation_mode="none", ooi=True, save=False)
        self.metric_list = self.metric_data.get_metrics()
        self._metric_by_name = {m.name: m for m in self.metric_list}
        self.aggregation_mode = self.metric_data.get_aggregation_mode()
        self.transformation_matrix = self.compute_alignment_matrix()
        self.score = self.metric_data.score

    def save_metric_data(self):
        self.metric_data.set_metrics(self.metric_list)
        self.metric_data.set_aggregation_mode(self.aggregation_mode)
        self.metric_data.save()

    def has_metric_data(self):
        return self.metric_data.has_save_data()

    def get_metric(self, metric_name: METRICS):
        return self._metric_by_name[metric_name]

    def compute_alignment_matrix(self):
        ooi_metric = self.get_metric(METRICS.OOI).measurement
        mask = ooi_metric == 1
        interesting_pts = self.points[mask]

        if len(interesting_pts) < 3:
            return np.eye(4)

        # PCA: dritte Hauptkomponente als Flächennormale
        pca = PCA(n_components=3)
        pca.fit(self.points)
        v3 = pca.components_[2]

        # Richtung prüfen – Normale soll zur Kamera zeigen
        camera_center = self.camera_poses[:, 9:].mean(axis=0)
        ooi_center = interesting_pts.mean(axis=0)
        direction_to_camera = camera_center - ooi_center

        if np.dot(v3, direction_to_camera) < 0:
            v3 = -v3

        # Rotation berechnen
        R = rotation_matrix_from_vectors(v3, np.array([0, 0, 1]))

        # Punkte rotieren und zentrieren
        centered = interesting_pts - ooi_center
        rotated = centered @ R.T

        # Skalierung
        scale = np.max(np.linalg.norm(rotated, axis=1))
        scale = scale if scale != 0 else 1.0
        rotated /= scale

        # Z-Verschiebung: Unterster Punkt auf Z=0
        z_min = rotated[:, 2].min()

        # Transformation kombinieren: zuerst rotieren & zentrieren
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ ooi_center

        # Dann skalieren
        S = np.eye(4)
        S[:3, :3] *= 2.0 / scale  # z.B. Normgröße = 2.0

        # Dann Verschiebung in Z-Richtung
        Z = np.eye(4)
        Z[2, 3] = -z_min  # nach oben verschieben

        # Gesamte Transformation: zuerst rotieren, dann skalieren, dann verschieben
        return Z @ S @ T

def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.allclose(c, 1):
        return np.eye(3)  # Keine Rotation nötig
    if np.allclose(c, -1):
        # 180° Rotation: wähle beliebige orthogonale Achse
        axis = np.eye(3)[np.argmin(np.abs(a))]
        return rotation_matrix_from_vectors(a, -axis) @ rotation_matrix_from_vectors(-axis, b)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (np.linalg.norm(v) ** 2))
    return R