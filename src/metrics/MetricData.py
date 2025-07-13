from enum import IntEnum
from pathlib import Path

import h5py
import numpy as np
from matplotlib import cm

from src.general.python.filePaths import ProjectPaths
from src.utils.JsonManager import JsonManager


class METRICS(IntEnum):

    CAMERA_DISTANCE = 1
    OBSERVATION_COUNT = 2
    TRIANGULATION_UNCERTAINTY = 3
    ANGLE_OF_INCIDENCE = 4
    PROJECTION_ERROR = 5
    REPROJECTION_ERROR = 6
    DENSITY = 7
    DISTANCE_TO_EDGE = 8
    EDGE_POINTS = 9
    COMBINED = 10
    BRIGHTNESS = 11
    DARKNESS = 12
    PRECISION = 13
    IMG_QUALITY = 14
    OOI = 15

class Metric:
    def __init__(self, metric: METRICS, measurement: np.ndarray, normalized: np.ndarray = None):
        if measurement is None:
            raise Exception("measurement is None")
        self.name = metric
        self.measurement = measurement
        self.min = np.min(measurement)
        self.max = np.max(measurement)
        self.display = self.measurement
        self.normalized = normalized

    def get_normalized(self):
        if self.normalized is None:
            print("normalizing metric")
            self.normalize()
        return self.normalized

    def convert_to_RGB(self, cmap: str = "jet"):
        cmap = cm.get_cmap(cmap)
        colors = cmap(self.display)
        colors[:, 3] = 1.0
        return colors

    def normalize(self):
        self.normalized = ((self.measurement - self.min) / (self.max - self.min)).reshape(-1,3)

    def set_normalized_display(self, normalized):
        self.normalized = normalized

class MetricData:
    def __init__(self, project_paths: ProjectPaths, metrics: list[Metric] | None = None, aggr_mode: str = None):
        if metrics is not None:
            self._metrics = metrics
        else:
            self._metrics = None

        self.aggregation_mode = aggr_mode
        self.project_paths = project_paths
        self._store_path = self.project_paths.data_path / "metrics.h5"
        self.score = None

    def set_metrics(self, metrics: list[Metric]):
        self._metrics = metrics

    def get_metrics(self) -> list[Metric]:
        return self._metrics

    def set_aggregation_mode(self, aggr_mode: str):
        self.aggregation_mode = aggr_mode

    def get_aggregation_mode(self) -> str:
        return self.aggregation_mode

    def get_metric(self, metric: METRICS) -> np.ndarray | None:
        for m in self._metrics:
            if m.name == metric:
                return m
        return None

    def save(self):
        print("saving data at: ", self._store_path)
        data = self._build_data_np()

        with h5py.File(self._store_path, "w") as file:
            for name, data_array in data.items():
                file.create_dataset(name, data=data_array, compression="gzip")



    def restore(self):
        print("restoring data from: ", self._store_path)
        if not self.has_save_data():
            print("no saved metric data found.")
            return

        metric_list = []
        with h5py.File(self._store_path, 'r') as file:
            for metric_key in METRICS:
                key = str(metric_key.value)
                measurement = file[key][:]
                normalized = file.get(key + "_norm")
                h5_metric = Metric(metric_key, measurement)
                if normalized is not None:
                    h5_metric.normalized = normalized[:]
                metric_list.append(h5_metric)
            if "aggregation_mode" in file:
                self.aggregation_mode = file["aggregation_mode"].asstr()[0]
            else:
                self.aggregation_mode = None
            if "score" in file:
                self.score = file["score"][0]
            else:
                self.score = None
        self._metrics = metric_list


    def export_to_json(self, export_path: Path = None, pointcloud: np.ndarray = None):
        if export_path is None:
            export_path = self.project_paths.export_path
            export_path.mkdir(parents=True, exist_ok=True)
        project_name = self.project_paths.project_root_path.name
        out = JsonManager(export_path, project_name + "_pm.json")
        data = self._build_data(pointcloud)
        out.update(data)

    def has_save_data(self):
        return self._store_path.exists()

    def _build_data_np(self):
        data = {}
        for metric in self._metrics:
            key = str(metric.name.value)
            data[key] = metric.measurement
            if metric.normalized is not None:
                data[key + "_norm"] = metric.normalized
        if self.aggregation_mode is not None:
            str_dtype = h5py.string_dtype()
            data["aggregation_mode"] = np.asarray([self.aggregation_mode], dtype=str_dtype)
        if self.score is not None:
            data["score"] = np.asarray([self.score], dtype=np.float32)
        return data

    def _build_data(self, pointcloud: np.ndarray = None):
        data = {}
        for metric in self._metrics:
            key = metric.name.name
            data[key] = metric.measurement.tolist()
            if metric.normalized is not None:
                data[key + "_NORM"] = metric.normalized.tolist()
        if self.aggregation_mode is not None:
            data["aggregation_mode"] = self.aggregation_mode
        if pointcloud is not None:
            data["POINT_CLOUD"] = pointcloud.tolist()
        return data