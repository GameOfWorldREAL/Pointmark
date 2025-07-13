from pathlib import Path
import tkinter as tk
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
from src.pointmark.setup.selectFunctions import select_video_path, select_keyframe_extractor, \
    select_photogrammetry_pipeline, select_project_path
from src.pointmark.setup.batch_setup import is_batch
from src.utils.setup_utils import build_empty_settings, print_settings_found
from src.utils.JsonManager import JsonManager
from src.utils.utils import path_to_str

#--------------------- UI ------------------
def print_pointmark_logo():
    print("  ___           _                  _   ")
    print(" | _ \\___(_)_ _| |_ _ __  __ _ _ _| |__")
    print(" |  _/ _ \\ | ' \\  _| '  \\/ _` | '_| / /")
    print(" |_| \\___/_|_||_\\__|_|_|_\\__,_|_| |_\\_\\")

def setup_UI():
    #------------------------- setup UI -----------------------------
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except(ImportError, AttributeError, OSError):
        pass

#--------------------- single pipeline functions -----------------
# initialize project
def init_pointmark(args, data_path, description: str, warning: str, pm_function, _FLAG_ONLY):
    pointmark_paths = PointmarkPaths(Path(__file__).resolve().parents[3])
    skip = args.yes
    metrics = args.metrics
    agg_mode = args.aggregation_mode
    ooi = args.ooi
    export = args.export

    if metrics is False:
        if ooi is True:
            raise Exception("'--ooi' can only be used with '--metrics'")
        if agg_mode is not None:
            raise Exception("'--aggregation_mode' can only be used with '--metrics'")
        if export is not None:
            raise Exception("'--export' can only be used with '--metrics'")

    # set a project path
    if args.project_path is None:
        print("=======================================")
        print(f"run setup: {description}")
        print("=======================================")
        project_path = select_project_path(pointmark_paths, skip).project_root_path
    else:
        project_path = args.project_path

    print("project:", project_path)

    if export is _FLAG_ONLY:
        export = ProjectPaths(project_path).export_path

    # data path can be None then it should already be computed
    if data_path is None:
        print(warning)
        pm_function(pointmark_paths, project_path, None, skip, metrics, agg_mode, ooi, export)
    else:
        pm_function(pointmark_paths, project_path, data_path, skip, metrics, agg_mode, ooi, export)

# restore and setup single pipeline settings
def setup_single_pipeline(settings_json: JsonManager, pointmark_paths: PointmarkPaths, project_paths: ProjectPaths,
                          video_path: Path=None, skip_video=False, skip=False):
    #restore settings if already saved
    if not settings_json.is_empty():
        if is_batch(settings_json):
            raise Exception("this pipeline does not support batch projects")

        if skip:
            print("settings restored")
            return

        print_settings_found(settings_json)
        choice = input("do you want to use saved settings press Y else press N:\n")
        if choice.upper() == "Y":
            print("settings restored")
            return

    settings_json.set_entry("batch", False)
    settings_json.set_entry("project_path", path_to_str(project_paths.project_root_path))

    #select settings and save them in the settings.json file
    if video_path is None:
        if not skip_video:
            select_video_path(settings_json, skip)
        else:
            settings_json.set_entry("video_path", None)
    else:
        settings_json.set_entry("video_path", path_to_str(video_path))

    if not skip_video:
        select_keyframe_extractor(settings_json)
    select_photogrammetry_pipeline(settings_json, pointmark_paths)
    print("setup finished")

# build settings for a project
def prepare_project(settings_json: JsonManager, project_paths: ProjectPaths, video_path: Path = None):
    settings = build_empty_settings()
    #setup project settings
    settings["batch"] = False
    settings["project_path"] = path_to_str(project_paths.project_root_path)
    if video_path is not None:
        settings["video_path"] = path_to_str(video_path)
    for key in settings:
        if settings[key] is None:
            settings[key] = settings_json.get_value(key)
    settings_json.update(settings)