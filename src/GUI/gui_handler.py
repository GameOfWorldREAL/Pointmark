import copy

import numpy as np
from pathlib import Path
from PySide6.QtWidgets import QFileDialog
from matplotlib import cm

from src.GUI.Project import Project
from src.general.python.filePaths import PointmarkPaths
from src.metrics.MetricData import METRICS
from src.metrics.compute_metrics import reaggregate
from src.utils.JsonManager import JsonManager

#=========================
#chatGPT 4.o generated gui
#=========================

pointmark_paths = PointmarkPaths(Path(__file__).resolve().parents[2])

def str_to_METRICS(s: str) -> METRICS:
    lookup = {
        "Camera Distance":           METRICS.CAMERA_DISTANCE,
        "Observation Count":         METRICS.OBSERVATION_COUNT,
        "Triangulation Uncertainty": METRICS.TRIANGULATION_UNCERTAINTY,
        "Projection Error":          METRICS.PROJECTION_ERROR,
        "Reprojection Error":        METRICS.REPROJECTION_ERROR,
        "Density":                   METRICS.DENSITY,
        "Angle of Incidence":        METRICS.ANGLE_OF_INCIDENCE,
        "Distance to Edge":          METRICS.DISTANCE_TO_EDGE,
        "Edge Points":               METRICS.EDGE_POINTS,
        "Combined":                  METRICS.COMBINED,
        "Brightness":                METRICS.BRIGHTNESS,
        "Darkness":                  METRICS.DARKNESS,
        "Image Quality":             METRICS.IMG_QUALITY,
        "Precision":                 METRICS.PRECISION,
        "OOI":                       METRICS.OOI
    }
    try:
        return lookup[s]
    except KeyError:
        raise ValueError(f"Unknown metric name: {s}")

def handle_open_project(window):
    Path(pointmark_paths.cache_path).mkdir(parents=True, exist_ok=True)
    project_json = JsonManager(pointmark_paths.cache_path, "project.json")

    # Lade letzten Projektpfad, oder nimm Root als Fallback
    init_path = "\\"
    if not project_json.is_empty():
        init_path = copy.copy(project_json.load().get("last_project_path", init_path))

    # Öffne Dialog mit vorausgewähltem Verzeichnis
    folder = QFileDialog.getExistingDirectory(
        window,
        "Projektordner auswählen",
        init_path
    )
    if folder:
        name = folder.split("/")[-1]
        project = Project(name, folder)
        window.left_dock.project_list.add_project(project)

        # Speichere den gewählten Pfad
        project_json.update({"last_project_path": folder})

def handle_project_selected(window, project):
    if window.current_project is not None:
        old_key = window.current_project.path.as_posix()
        window._camera_states[old_key] = window.viewer.plotter.camera_position

    window.current_project = project

    # Aggregationsmodus aus Projekt in UI übernehmen
    agg_mode = getattr(project, "aggregation_mode", "none")
    window.right_dock.aggregation_combo.setCurrentText(agg_mode)

    window._pv_updater.update_from_project(
        project,
        metric_name=window.display_params["metric"],
        point_size=window.display_params["point_size"],
        color_scale=window.display_params["color_scale"],
        show_cameras=window.display_params.get("show_cameras", "None"),
        frustum_scale=window.display_params["frustum_scale"],
        display_grid=window.display_params.get("display_grid", False),
        ooi_alpha=window.display_params.get("ooi_alpha", 0.1),
        reset_camera=True
    )

    new_key = project.path.as_posix()
    cam = window._camera_states.get(new_key)
    if cam is not None:
        window.viewer.plotter.camera_position = cam
    else:
        window.viewer.plotter.reset_camera()
    window.viewer.plotter.render()


def handle_display_parameters_changed(window, params: dict):
    new_mode = params.get("aggregation_mode")
    current_project = window.current_project

    # Nichts tun, wenn kein Projekt geladen
    if current_project is None:
        return

    prev_mode = getattr(current_project, "aggregation_mode", "none")

    # Nur neu aggregieren, wenn sich der Aggregationsmodus WIRKLICH geändert hat
    if new_mode != prev_mode:
        print(f"[INFO] Aggregation mode changed: {prev_mode} -> {new_mode}")
        if not current_project.has_metric_data():
            current_project.save_metric_data()
        metric_list = reaggregate(
            pointmark_paths,
            current_project.project_paths,
            new_mode
        )
        current_project.metric_data.set_metrics(metric_list)
        current_project.metric_list = metric_list
        current_project._metric_by_name = {m.name: m for m in metric_list}
        current_project.aggregation_mode = new_mode

    # 🟡 HIER: Anzeige-Parameter aktualisieren
    window.display_params.update(params)

    alpha_val = params.get("ooi_alpha", 10)
    # Viewer aktualisieren
    window._pv_updater.update_from_project(
        window.current_project,
        metric_name=window.display_params["metric"],
        point_size=window.display_params["point_size"],
        color_scale=window.display_params["color_scale"],
        show_cameras=window.display_params.get("show_cameras", "None"),
        frustum_scale=window.display_params["frustum_scale"],
        display_grid=window.display_params["display_grid"],
        reset_camera=False,
        ooi_alpha=alpha_val / 100.0  # als [0.0, 1.0]
    )



class PyVistaUpdater:
    def __init__(self, viewer):
        self._viewer = viewer

    def update_from_project(
            self,
            project: Project,
            *,
            metric_name: str = "Combined",
            point_size: int = 5,
            color_scale: str = "jet",
            show_cameras: str = "None",
            frustum_scale: float = 0.05,
            reset_camera: bool = False,
            display_grid: bool = False,
            ooi_alpha: float = 0.6
    ):
        # 1) Punktwolke vorbereiten
        pts = apply_transform(project.points, project.transformation_matrix)
        metric = project.get_metric(str_to_METRICS(metric_name))
        scalars = metric.get_normalized()
        ooi = project.get_metric(METRICS.OOI).measurement
        alpha_array = np.where(ooi == 1, 1.0, ooi_alpha)

        # 2) Punktwolke anzeigen (mit Alpha)
        self._viewer.load_point_cloud(
            pts,
            scalars=scalars,
            point_size=point_size,
            cmap=color_scale,
            brightness=0.85,
            alpha_array=alpha_array,
            reset_camera=reset_camera
        )

        # 3) Kamerafarbe (kontrastreich zur Colormap und Hintergrund)
        mid_rgb = np.array(cm.get_cmap(color_scale)(0.5)[:3], dtype=np.float32)
        comp_mid = 1.0 - mid_rgb
        bg = np.array(self._viewer.background_color, dtype=np.float32)
        comp_bg = 1.0 - bg
        cam_color = tuple(comp_mid) if np.linalg.norm(comp_mid - bg) > np.linalg.norm(comp_bg - bg) else tuple(comp_bg)

        # 4) Kamera-Transformationen
        cams = transform_camera_poses(project.camera_poses, project.transformation_matrix)

        # 4.1) Skalenfreie Matrix nur für Frustums (Größe soll gleich bleiben)
        R = project.transformation_matrix[:3, :3]
        scale = np.cbrt(np.linalg.det(R)) if np.linalg.det(R) > 0 else 1.0

        # 5) Kameras/Frustums anzeigen
        if show_cameras == "Show Cameras":
            self._viewer.add_camera_points(cams, color=cam_color, point_size=point_size * 2)
        elif show_cameras == "Show Frustum":
            self._viewer.add_camera_points(cams, color=cam_color, point_size=point_size * 2)
            self._viewer.add_camera_frustums(
                cams,
                scale=frustum_scale / scale,
                color=cam_color,
                line_width=2
            )

        # 6) Grid anzeigen
        self._viewer.update_grid(display_grid)

        # 7) Abschließend rendern
        self._viewer.plotter.render()


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """ wendet eine 4x4-Transformationsmatrix auf Nx3 Punkte an """
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # → Nx4
    transformed = points_h @ matrix.T
    return transformed[:, :3]

import numpy as np

def transform_camera_poses(camera_poses: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transformed = []
    for cam in camera_poses:
        R = cam[:9].reshape(3, 3)
        C = cam[9:].reshape(3)

        R_new = transform[:3, :3] @ R
        C_new = transform[:3, :3] @ C + transform[:3, 3]

        cam_new = np.concatenate([R_new.flatten(), C_new])
        transformed.append(cam_new)
    return np.array(transformed)
