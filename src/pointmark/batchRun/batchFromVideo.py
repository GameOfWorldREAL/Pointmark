import sys
from pathlib import Path
pointmark_path = Path(__file__).resolve().parent
src_path = str(pointmark_path / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.general.python.cmdRunner import run_keyframe_selection
from src.general.python.dirBuilder import build_batch_videos_dir
from src.general.python.filePaths import BatchPaths, PointmarkPaths, ProjectPaths
from src.pointmark.batchRun.batchFromImages import batchFromImage
from src.pointmark.setup.batch_setup import setup_batch_pipeline, build_sub_project, build_sub_project_settings, build_setup_json
from src.utils.setup_utils import build_empty_batch_settings
from src.utils.JsonManager import JsonManager
from src.utils.utils import copy_videos, detect_videos, path_to_str

#set up the batch pipeline
def _setup_batch_pipeline(batch_stg_json: JsonManager,pointmark_paths: PointmarkPaths, project_path: Path,
                          videos_path: Path, setup_path=None, skip: bool=False):
    print("=======================================")
    print("detected videos:")
    det_videos = detect_videos(videos_path, print_out=True)
    print("=======================================")
    batch_paths = BatchPaths(project_path)
    data_json = JsonManager(batch_paths.data_path, "data.json")
    data_json.set_entry("videos", det_videos)

    setup_batch_pipeline(batch_stg_json, pointmark_paths, project_path, setup_path, skip=skip)

#build the subprojects and their settings for the batch project folder
def _build_project(batch_stg_json: JsonManager, video_paths, setup: dict=None):
    batch_stg = batch_stg_json.load()
    batch_paths = BatchPaths(Path(batch_stg["project_path"]))
    batch_project_paths = []
    for video in video_paths:
        project_name = Path(video).stem
        settings, project_paths = build_sub_project(batch_paths, project_name, setup)

        settings["video_path"] = video
        build_sub_project_settings(batch_stg_json, project_paths, settings)

        batch_project_paths.append(path_to_str(project_paths.project_root_path))
    return batch_project_paths

#run keyframe extraction on each project in the batch project folder
def _batch_keyframe_extraction(pointmark_paths: PointmarkPaths, batch_project_paths):
    image_sets = []
    for project in batch_project_paths:
        print("processing images for project:", project)
        project_paths = ProjectPaths(Path(project))
        settings_json = JsonManager(project_paths.settings_path, "settings.json")
        run_keyframe_selection(settings_json, pointmark_paths, project_paths)
        image_sets.append(path_to_str(project_paths.images_path))
    return image_sets


def batchFromVideo(
        pointmark_paths: PointmarkPaths, project_path: Path = None, videos_path: Path = None,
        setup_path: Path = None, skip=False, metrics=False, aggregation_mode="min", ooi=True, export=False):
    """
    Processes video files and sets up a batch pipeline for keyframe extraction and further processing.
    This function handles copying videos, setting up configurations, and creating projects from the
    videos to enable large-scale image batch processing. Starts batchFromImage pipeline.

    :param export:
    :param ooi:
    :param aggregation_mode:
    :param metrics:
    :param pointmark_paths: Object containing paths related to pointmark configurations.
    :type pointmark_paths: PointmarkPaths
    :param project_path: Optional base path for the project, used to determine where batch data will be stored.
    :type project_path: Path, optional
    :param videos_path: Optional path to the video files to be included in the batch. If not provided,
                        videos will be copied into the batch's default video directory.
    :type videos_path: Path, optional
    :param setup_path: Optional path to the setup configuration JSON file. If not provided, the
                       default setup will be used.
    :type setup_path: Path, optional
    :param skip: Boolean indicating whether certain steps (such as keyframe extraction) should be skipped in the
                 pipeline. Defaults to False.
    :type skip: bool, optional
    :return: None
    """
    batch_paths = BatchPaths(project_path)
    build_batch_videos_dir(batch_paths)
    if not videos_path is None:
        if not videos_path is batch_paths.videos_path:
            copy_videos(videos_path, batch_paths.videos_path)
    else:
        videos_path = batch_paths.videos_path

    #setup
    batch_stg_json = JsonManager(batch_paths.settings_path, "batch_settings.json")
    if batch_stg_json.is_empty() and batch_stg_json.get_value("setup") is None:
        batch_stg_json.update(build_empty_batch_settings())
        _setup_batch_pipeline(batch_stg_json, pointmark_paths, project_path, videos_path, setup_path, skip)

    #build projects
    data_json = JsonManager(batch_paths.data_path, "data.json")
    data = data_json.load()
    video_paths = data["videos"]

    #batch mode
    if setup_path is None:
        batch_project_paths = _build_project(batch_stg_json, video_paths)
    #batch_setup mode
    else:
        setup = build_setup_json(setup_path)
        batch_project_paths = _build_project(batch_stg_json, video_paths, setup)

    data_json.set_entry("projects", batch_project_paths)
    image_sets = _batch_keyframe_extraction(pointmark_paths, batch_project_paths)
    data_json.set_entry("image_sets", image_sets)
    batchFromImage(pointmark_paths, project_path, skip=skip, metrics=metrics, aggregation_mode=aggregation_mode,
                   ooi=ooi, export=export)