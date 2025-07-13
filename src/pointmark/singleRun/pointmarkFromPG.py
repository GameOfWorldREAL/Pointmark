import sys
from pathlib import Path
pointmark_path = Path(__file__).resolve().parent
src_path = str(pointmark_path / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.ReconstructionData.Adapter.adapterUtils import copy_sfm_data
from src.ReconstructionData.ReconstructionData import ReconstructionData
from src.general.python.dirBuilder import build_pointmark_dir
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
from src.metrics.MetricData import MetricData
from src.metrics.compute_metrics import build_metrics
from src.pointmark.setup.setup import setup_single_pipeline, prepare_project
from src.utils.JsonManager import JsonManager
from src.utils.setup_utils import print_settings_found


def pointmarkFromPG(pointmark_paths: PointmarkPaths, project_path: Path, sfm_data_path: Path = None, skip=False,
                    metrics=False, aggregation_mode="min", ooi=True, export: Path = None):
    print(project_path)
    project_paths = ProjectPaths(project_path)
    build_pointmark_dir(project_paths)
    settings_json = JsonManager(project_paths.settings_path, "settings.json")
    if not sfm_data_path is None:
        print_settings_found(settings_json)
        setup_single_pipeline(settings_json, pointmark_paths, project_paths, skip_video=True, skip=skip)
        copy_sfm_data(int(settings_json.get_value("reconst_pipeline")), sfm_data_path, project_paths)
        prepare_project(settings_json, project_paths)

    print("prepare data")
    ReconstructionData(pointmark_paths, project_paths, silent_print=True)
    print("-----------------------------------------")


    if metrics:
        metric_data = MetricData(project_paths)
        print("build metrics")
        build_metrics(metric_data, pointmark_paths, aggregation_mode=aggregation_mode, ooi=ooi, save=True)
        print("-----------------------------------------")
        if export is not None:
            reconst_data = ReconstructionData(pointmark_paths, project_paths, silent_print=True)
            metric_data.export_to_json(pointcloud=reconst_data.get_point_cloud())
            project_name = project_paths.project_root_path.name
            print("export metrics to:", export / (project_name + "_pm.json"))