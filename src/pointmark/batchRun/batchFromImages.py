import sys
from pathlib import Path
pointmark_path = Path(__file__).resolve().parent
src_path = str(pointmark_path / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.general.python.cmdRunner import run_photogrammetry
from src.general.python.dirBuilder import build_batch_dir
from src.general.python.filePaths import PointmarkPaths, BatchPaths, ProjectPaths
from src.pointmark.batchRun.batchFromPG import batchFromPG
from src.pointmark.setup.batch_setup import build_setup_json, build_sub_project_settings, build_sub_project, setup_batch_pipeline
from src.utils.setup_utils import build_empty_batch_settings
from src.utils.JsonManager import JsonManager
from src.utils.utils import copy_images, image_ext, path_to_str

#returns a list of image sets; each image set is a folder containing images of the same scene.
def _get_image_set_paths(batch_images_dir: Path):
    image_sets = set()
    for file in batch_images_dir.iterdir():
        for image in file.iterdir():
            if image.suffix.lower() in image_ext:
                image_sets.add(path_to_str(file))
    return list(image_sets)

#set up the batch pipeline
def _setup_batch_pipeline(batch_stg_json: JsonManager, pointmark_paths: PointmarkPaths, project_path: Path, image_sets,
                          setup_path: Path=None, skip=False):
    print("=======================================")
    print("detected image sets:")
    for image_set in image_sets:
        print(Path(image_set).name + ":")
        print(image_set)
    print("=======================================")
    setup_batch_pipeline(batch_stg_json, pointmark_paths, project_path, setup_path, skip=skip)

#run photogrammetry on each project in the batch project folder
def _batch_run_photogrammetry(batch_stg_json: JsonManager, pointmark_paths: PointmarkPaths):
    batch_paths = BatchPaths(Path(batch_stg_json.get_value("project_path")))
    batch_project_paths = JsonManager(batch_paths.data_path, "data.json").get_value("projects")

    for project in batch_project_paths:
        try:
            print("project:", project)
            print("processing photogrammetry")
            project_paths = ProjectPaths(Path(project))
            settings_json = JsonManager(project_paths.settings_path, "settings.json")
            run_photogrammetry(settings_json, pointmark_paths, project_paths)
            print("-----------------------------------------")

        except Exception as e:
            print("error during photogrammetry processing:", e)
            print("-----------------------------------------")
            continue


#build the subprojects and their settings for the batch project folder
def _build_projects(batch_stg_json: JsonManager, image_sets, setup: dict = None):
    batch_stg = batch_stg_json.load()
    batch_paths = BatchPaths(Path(batch_stg["project_path"]))

    batch_project_paths = []
    project_image_sets = []
    for set_path in image_sets:
        project_name = Path(set_path).name
        settings, project_paths = build_sub_project(batch_paths, project_name, setup)
        copy_images(Path(set_path), project_paths.images_path)
        build_sub_project_settings(batch_stg_json, project_paths, settings)

        batch_project_paths.append(path_to_str(project_paths.project_root_path))
        project_image_sets.append(path_to_str(project_paths.images_path))
    return batch_project_paths, project_image_sets

def batchFromImage(pointmark_paths: PointmarkPaths, project_path: Path, image_sets_path: Path = None,
                   setup_path: Path = None, skip=False, metrics=False, aggregation_mode="min", ooi=True, export=False):
    """
    Processes a batch of image sets for photogrammetry and related tasks, handling setup and project-specific
    configurations. The function leverages JSON configurations to manage batch settings and data, including
    image sets and project paths. Starts batchFromPG pipeline.

    :param export:
    :param ooi:
    :param aggregation_mode:
    :param metrics:
    :param pointmark_paths: Required paths for pointmark operations.
    :type pointmark_paths: PointmarkPaths
    :param project_path: Path to the project directory.
    :type project_path: Path
    :param image_sets_path: Optional path to the directory containing image sets.
    :type image_sets_path: Path, optional
    :param setup_path: Optional path to the setup configuration file.
    :type setup_path: Path, optional
    :param skip: Flag to enable or disable questions during setup.
    :type skip: bool
    :return: None
    """
    batch_paths = BatchPaths(project_path)
    build_batch_dir(batch_paths)
    batch_stg_json = JsonManager(batch_paths.settings_path, "batch_settings.json")

    #configure the batch pipeline and copy the images to the corresponding project folder
    if image_sets_path is not None:
        image_sets = _get_image_set_paths(image_sets_path)
        if batch_stg_json.is_empty() and batch_stg_json.get_value("setup") is None:
            batch_stg_json.update(build_empty_batch_settings())
            _setup_batch_pipeline(batch_stg_json, pointmark_paths, project_path, image_sets, setup_path, skip)

        #batch mode
        if setup_path is None:
            batch_project_paths, project_image_sets = _build_projects(batch_stg_json, image_sets)
        #batch_setup mode
        else:
            setup = build_setup_json(setup_path)
            batch_project_paths, project_image_sets = _build_projects(batch_stg_json, image_sets, setup)

        data_json = JsonManager(batch_paths.data_path, "data.json")
        data_json.set_entry("projects", batch_project_paths)
        data_json.set_entry("image_sets", project_image_sets)

    _batch_run_photogrammetry(batch_stg_json, pointmark_paths)
    batchFromPG(pointmark_paths, project_path=project_path, skip=skip, metrics=metrics, aggregation_mode=aggregation_mode,
                ooi=ooi, export=export)