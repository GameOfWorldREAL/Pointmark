import copy
from pathlib import Path
from tkinter import filedialog
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
from src.utils.JsonManager import JsonManager
from src.utils.utils import enter_to_continue, ask_to_exit, is_integer, path_to_str

# select functions for setup

# select the project path
def select_project_path(pointmark_paths: PointmarkPaths, skip=False):
    # get last project path
    project_json = JsonManager(pointmark_paths.cache_path, "project.json")

    # init path for easy file search
    init_path = "\\"
    if not project_json.is_empty():
        init_path = copy.copy(project_json.load()["last_project_path"])

    print("please select project path")
    enter_to_continue(skip)

    project_path = None

    # select a project path with file dialog
    while not project_path:
        project_path = filedialog.askdirectory(
            title="select project path",
            initialdir=init_path)
        if not project_path:
            print("no project path selected")
            ask_to_exit(skip)

    project_path = path_to_str(project_path)
    project_json.update({"last_project_path": project_path})
    print("project path: ", project_path)
    return ProjectPaths(Path(project_path))

# select the video path for the project
def select_video_path(settings_json: JsonManager, skip=False):
    # set a default path to a project path
    project_path = Path(settings_json.get_value("project_path"))
    video_path = settings_json.get_value("video_path")

    #init path for easy file search
    init_path = project_path
    if not video_path is None and video_path.exists():
        init_path = Path(copy.copy(video_path)).parent

    # select a video path with file dialog
    video_path = None
    print("please select a video file")
    enter_to_continue()
    while not video_path:
        video_path = filedialog.askopenfilename(
            title="select Video File",
            filetypes=(("Video Files", "*.mp4"), ("All Files", "*.*")),
            initialdir=init_path
        )
        if not video_path:
            print("No video file selected")
            ask_to_exit(skip)
    video_path = path_to_str(video_path)
    print("video path: ", video_path)
    settings_json.set_entry("video_path", video_path)

# select the keyframe extractor for the project
def select_keyframe_extractor(settings_json: JsonManager):
    print("please choose your keyframe extractor:")
    print("1) jkeyframer")

    keyframer = 0
    while keyframer == 0:
        keyframer = input("type number:\n")
        if not is_integer(keyframer):
            keyframer = 0
            print("no valid input")
            ask_to_exit()
            continue
        if not (0 < int(keyframer) <= 1):  # set num of Extractors here
            keyframer = 0
            print("no valid input")
            ask_to_exit()
    settings_json.set_entry("keyframer", keyframer)

# select the photogrammetry pipeline for the project
def select_photogrammetry_pipeline(settings_json: JsonManager, pointmark_paths: PointmarkPaths):
    project_path = Path(settings_json.get_value("project_path"))

    print("please choose your photogrammetry pipeline:")
    print("1) Meshroom")
    print("2) Colmap (No Benchmark Support)")
    reconst_pipeline = 0
    while reconst_pipeline == 0:
        reconst_pipeline = input("type number:\n")
        if not is_integer(reconst_pipeline):
            reconst_pipeline = 0
            print("no valid input")
            ask_to_exit()
            continue
        if not (0 < int(reconst_pipeline) <= 2):       # set num of Photogrammetry Software here
            reconst_pipeline = 0
            print("no valid input")
            ask_to_exit()

    #set up Meshroom
    if int(reconst_pipeline) == 1:
        meshroom_path = settings_json.get_value("meshroom_path")
        template_path = settings_json.get_value("template_path")

        # init path for easy file search
        init_path1 = project_path
        if not meshroom_path is None and Path(meshroom_path).exists():
            init_path1 = copy.copy(meshroom_path)

        init_path2 = project_path
        if not template_path is None and Path(template_path).exists():
            init_path2 = Path(copy.copy(template_path)).parent

        print("please select your Meshroom folder")
        input("press enter to continue...")
        meshroom_path = filedialog.askdirectory(
            title="select photogrammetry pipeline",
            initialdir=init_path1
        )
        print("please select a Template or use standard Template")
        choice = input("do you want to use a standard Template press Y else press N:\n")
        if not choice.upper() == "Y":
            template_path = filedialog.askopenfilename(
                title="select template",
                filetypes=(("Template Files", "*.mg"), ("All Files", "*.*")),
                initialdir=init_path2
            )
        else:
            template_path = str(pointmark_paths.templates_path / "meshroom.mg")

        template_path = path_to_str(template_path)
        meshroom_path = path_to_str(meshroom_path)
        settings_json.set_entry("template_path", template_path)
        settings_json.set_entry("meshroom_path", meshroom_path)
        settings_json.set_entry("reconst_pipeline", reconst_pipeline)


    #setup Colmap
    elif int(reconst_pipeline) == 2:
        colmap_path = Path(settings_json.get_value("colmap_path"))

        init_path = project_path
        if not colmap_path is None and colmap_path.exists():
            init_path = copy.copy(colmap_path)

        print("please select your Colmap folder")
        input("press enter to continue...")
        colmap_path = filedialog.askdirectory(
            title="select photogrammetry pipeline",
            initialdir=init_path
        )
        colmap_path = path_to_str(colmap_path)
        settings_json.set_entry("colmap_path", colmap_path)
    else:
        raise Exception("Photogrammetry pipeline not existing")