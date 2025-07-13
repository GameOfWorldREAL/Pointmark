import sys
from pathlib import Path
pointmark_path = Path(__file__).resolve().parent
src_path = str(pointmark_path / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.ReconstructionData.Adapter.adapterUtils import copy_sfm_data
from src.ReconstructionData.ReconstructionData import ReconstructionData
from src.general.python.dirBuilder import build_batch_dir
from src.general.python.filePaths import PointmarkPaths, ProjectPaths, BatchPaths
from src.metrics.MetricData import MetricData
from src.metrics.compute_metrics import build_metrics
from src.pointmark.setup.batch_setup import build_sub_project_settings, build_setup_json, build_sub_project, setup_batch_pipeline
from src.utils.setup_utils import build_empty_batch_settings
from src.utils.JsonManager import JsonManager
from src.utils.utils import path_to_str

#returns a list of sfm_data sets; each sfm_data set is a folder containing sfm.json.
def get_sfm_data_paths(batch_sfm_data_dir:Path):
    sfm_data = set()
    for file in batch_sfm_data_dir.iterdir():
        sfm_data.add(path_to_str(file))
    return list(sfm_data)

#set up the batch pipeline
def _setup_batch_pipeline(batch_stg_json: JsonManager, pointmark_paths: PointmarkPaths, project_path: Path, sfm_data_sets,
                          setup_path: Path=None, skip=False):
    print("=======================================")
    print("detected sfm_data sets:")
    for sfm_data_sets in sfm_data_sets:
        print(Path(sfm_data_sets).name + ":")
        print(sfm_data_sets)
    print("=======================================")
    setup_batch_pipeline(batch_stg_json, pointmark_paths, project_path, setup_path, skip=skip)

#build the subprojects and their settings for the batch project folder
def _build_projects(batch_stg_json: JsonManager, sfm_data_sets, setup=None):
    print("building projects")
    batch_stg = batch_stg_json.load()
    batch_paths = BatchPaths(Path(batch_stg["project_path"]))

    batch_project_paths = []
    project_image_sets = []
    for set_path in sfm_data_sets:
        project_name = Path(set_path).name
        settings, project_paths = build_sub_project(batch_paths, project_name, setup)

        #batch_setup mode
        if setup is not None:
            copy_sfm_data(int(settings["reconst_pipeline"]), Path(set_path), project_paths)
        #batch mode
        else:
            copy_sfm_data(int(batch_stg_json.get_value("reconst_pipeline")), Path(set_path), project_paths)

        build_sub_project_settings(batch_stg_json, project_paths, settings)

        batch_project_paths.append(path_to_str(project_paths.project_root_path))
        project_image_sets.append(path_to_str(project_paths.images_path))
    return batch_project_paths, project_image_sets

def batchFromPG(pointmark_paths: PointmarkPaths, project_path: Path, sfm_data_path: Path = None,
                setup_path: Path=None, skip=False, metrics=False, aggregation_mode="min", ooi=True, export=True):
    """
    Processes batch data from given paths and settings, handling reconstruction, project configurations, and generating
    associated metadata. Starts the pointmark quality analysis pipeline.

    :param export:
    :param ooi:
    :param aggregation_mode:
    :param metrics:
    :param pointmark_paths: Paths related to pointmark data. Drives processing of
                            pointmark quality evaluations.
    :type pointmark_paths: PointmarkPaths
    :param project_path: Root path of the current project. Used to manage batch
                         data paths and project settings.
    :type project_path: Path
    :param sfm_data_path: Path to structure-from-motion (sfm) datasets. Includes
                          data necessary for reconstructing project information.
                          Optional parameter.
    :type sfm_data_path: Path, optional
    :param setup_path: Path to setup configurations. Provides additional setup
                       details for project handling and builds if available.
                       Optional parameter.
    :type setup_path: Path, optional
    :param skip: Flag to indicate whether specific steps should be skipped
                 during processing. Default is False.
    :type skip: bool
    :return: None
    """
    batch_paths = BatchPaths(project_path)
    build_batch_dir(batch_paths)
    batch_stg_json = JsonManager(batch_paths.settings_path, "batch_settings.json")
    if  sfm_data_path is not None:
        sfm_data_sets = get_sfm_data_paths(sfm_data_path)
        if batch_stg_json.is_empty() and batch_stg_json.get_value("setup") is None:
            batch_stg_json.update(build_empty_batch_settings())
            _setup_batch_pipeline(batch_stg_json, pointmark_paths, project_path, sfm_data_sets, setup_path, skip)
        if setup_path is None:
            batch_project_paths, project_image_sets = _build_projects(batch_stg_json, sfm_data_sets)
        else:
            setup = build_setup_json(setup_path)
            batch_project_paths, project_image_sets = _build_projects(batch_stg_json, sfm_data_sets, setup)

        data_json = JsonManager(batch_paths.data_path, "data.json")
        data_json.set_entry("projects", batch_project_paths)
        data_json.set_entry("image_sets", project_image_sets)

    print("Pointmark quality analysis:")
    print("PointmarkPaths:", pointmark_paths.pointmark_root_path)
    print("ProjectPath:", project_path)
    print("-----------------------------------------")
    data_json = JsonManager(batch_paths.data_path, "data.json")
    projects = data_json.get_value("projects")

    for project in projects:
        try:
            project_paths = ProjectPaths(Path(project))
            print("project:", project)
            print("prepare data")
            ReconstructionData(pointmark_paths, project_paths, silent_print=True)
            print("-----------------------------------------")
        except Exception as e:
            print("error during data preparation:", e)
            print("-----------------------------------------")
            continue

    #compute metrics
    if not metrics:
        return

    for project in projects:
        project_paths = ProjectPaths(Path(project))
        metric_data = MetricData(project_paths)
        print("project:", project)
        print("build metrics")
        build_metrics(metric_data, pointmark_paths, aggregation_mode, ooi, True)
        if export:
            reconst_data = ReconstructionData(pointmark_paths, project_paths, silent_print=True)
            metric_data.export_to_json(pointcloud=reconst_data.get_point_cloud())
            project_name = project_paths.project_root_path.name
            print("export metrics to:", project_paths.export_path / (project_name + "_pm.json"))
        print("-----------------------------------------")