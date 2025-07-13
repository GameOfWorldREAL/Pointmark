from pathlib import Path

from src.general.python.dirBuilder import build_pointmark_dir
from src.general.python.filePaths import ProjectPaths, BatchPaths, PointmarkPaths
from src.pointmark.setup.selectFunctions import select_photogrammetry_pipeline, select_keyframe_extractor, \
    select_project_path
from src.utils.setup_utils import build_empty_settings, is_batch
from src.utils.JsonManager import JsonManager
from src.utils.utils import path_to_str

#--------------------- batch setup functions -----------------
# initialize batch project
def init_batch(args, data_path, description: str, warning: str, batch_function):
    pointmark_paths = PointmarkPaths(Path(__file__).resolve().parents[3])
    metrics = args.metrics
    agg_mode = args.aggregation_mode
    ooi = args.ooi
    export = args.export

    if metrics is False:
        if ooi is True:
            raise Exception("--ooi can only be used with metrics")
        if agg_mode is not None:
            raise Exception("--aggregation_mode can only be used with metrics")
        if export is True:
            raise Exception("--export can only be used with metrics")

    # set a project path
    if args.project_path is None:
        print("=======================================")
        print(f"run setup: {description}")
        print("=======================================")
        project_path = select_project_path(pointmark_paths, True).project_root_path
    else:
        project_path = args.project_path

    print("batch project:", project_path)

    # data path can be None then it should already be computed
    setup_path = args.setup_path

    if data_path is None:
        print(warning)
        batch_function(pointmark_paths, project_path, None, setup_path, True, metrics, agg_mode, ooi, export)
    else:
        batch_function(pointmark_paths, project_path, data_path, setup_path, True, metrics, agg_mode, ooi, export)

# check if batch settings are already saved and ask if they should be restored
def _restore_batch_settings(batch_stg_json: JsonManager, skip=False):
    if not batch_stg_json.is_empty():
        if not is_batch(batch_stg_json):
            raise Exception("this pipeline is only for batch projects")
        print("settings found:")
        print("---------------------------------------")
        print(batch_stg_json)
        print("---------------------------------------")
        if skip:
            print("settings restored")
            return True
        choice = input("do you want to use saved settings press Y else press N:\n")
        if choice.upper() == "Y":
            print("settings restored")
            return True
    return False

# select batch settings and save them in the batch_settings.json file
def setup_batch_pipeline(batch_stg_json: JsonManager, pointmark_paths: PointmarkPaths, project_path: Path, setup_path: Path=None, skip=False):
    batch_paths = BatchPaths(project_path)
    data_json = JsonManager(batch_paths.data_path, "data.json")
    # check if batch_setup mode is used
    if (setup_path is not None) or (batch_stg_json.get_value("setup") is not None):
        if setup_path is not None:
            batch_stg_json.set_entry("setup", path_to_str(setup_path))
        batch_stg_json.set_entry("batch", True)
        batch_stg_json.set_entry("project_path", path_to_str(project_path))
        batch_stg_json.set_entry("data_path", path_to_str(batch_paths.data_path / data_json.filename))
    else:
        # if batch mode is used, try to restore settings or run setup
        if not _restore_batch_settings(batch_stg_json, skip):
            batch_stg_json.set_entry("batch", True)
            batch_stg_json.set_entry("project_path", path_to_str(project_path))
            batch_stg_json.set_entry("data_path", path_to_str(batch_paths.data_path / data_json.filename))
            select_keyframe_extractor(batch_stg_json)
            select_photogrammetry_pipeline(batch_stg_json, pointmark_paths)

#restructure the batch_setup.json file to a dictionary of projects and their settings
def build_setup_json(setup_path: Path):
    batch_setup = JsonManager(setup_path, "batch_setup.json").load()
    setup = {}
    for settings, projects in batch_setup.items():
        for project in projects:
            setup[project] = (setup_path / settings).with_suffix(".json")
    return setup

#build settings shell based on pre-defined setup for a subproject of batch_setup mode
def _get_setup_settings(project_name: str, setup) -> dict:
    #load setup values
    settings = build_empty_settings()
    print(project_name)
    if not project_name in setup:
        raise Exception(f"Project {project_name} not found in setup.json")
    else:
        settings_path = setup[project_name]

    #add settings keys to settings if not already present
    setup_settings_json = JsonManager(settings_path.parent, settings_path.name)
    setup_settings = setup_settings_json.load()
    for key in settings:
        if key not in setup_settings:
            setup_settings[key] = None
    # return settings shell with setup values
    return setup_settings

# build a subproject of a batch project
def build_sub_project(batch_paths: BatchPaths, project_name: str, setup):
    print("Building project: " + project_name)
    # get settings shell to be used
    settings = build_empty_settings()
    if setup is not None:
        settings = _get_setup_settings(project_name, setup)

    #set a project path and build its directory structure
    new_project_path = batch_paths.projects_path / project_name
    new_project_path.mkdir(parents=True, exist_ok=True)

    project_paths = ProjectPaths(new_project_path)
    build_pointmark_dir(project_paths)
    return settings, project_paths

# build subproject settings based on prepared settings shell
def build_sub_project_settings(batch_stg_json: JsonManager, project_paths: ProjectPaths, settings):
    settings_json = JsonManager(project_paths.settings_path, "settings.json")
    # setup project settings
    settings["batch"] = False
    settings["project_path"] = path_to_str(project_paths.project_root_path)
    for key in settings:
        if settings[key] is None:
            settings[key] = batch_stg_json.get_value(key)
    settings_json.update(settings)