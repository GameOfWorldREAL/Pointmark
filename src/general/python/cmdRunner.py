import subprocess
from pathlib import Path
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
from src.general.python.dirBuilder import build_pg_dir
from src.utils.JsonManager import JsonManager
from src.utils.utils import path_to_str


#------------------ run keyframe selection ---------------------
#runs the keyframe selection on the video and settings provided in the project
def run_keyframe_selection(settings_json: JsonManager, pointmark_paths: PointmarkPaths, project_paths: ProjectPaths):
    if settings_json.get_value("keyframer") is None:
        raise Exception("keyframer not set")

    settings = settings_json.load()
    #check if images exist
    images_path = project_paths.images_path
    if images_path.exists() and any(images_path.iterdir()):
        print("Images already exists: " + path_to_str(images_path))
        return
    #keyframeSelection.cmd will manage all computation steps, because keyframer are build in
    if int(settings["keyframer"]) == 1:
        print("start jkeyframer")
        print("a folder \"images\" will be created in your project folder")
    #keyframeSelection.cmd expects: path to keyframers, path to video, path to project, keyframer number
    elif int(settings["keyframer"]) == 2:
        print("start keyframeSelection.cmd")
    subprocess.run(
        [
            pointmark_paths.keyframe_selection_cmd,
            pointmark_paths.keyframe_selector_path,
            settings["video_path"],
            settings["project_path"],
            settings["keyframer"]
        ],
        shell=True
    )

#------------------ run photogrammetry ---------------------
#runs the photogrammetry on the images and settings provided in the project
def run_photogrammetry(settings_json: JsonManager, pointmark_paths: PointmarkPaths, project_paths: ProjectPaths):
    #check if a photogrammetry pipeline is set
    # -> most likely not set because batch_setup (multi settings) used in batch (single setting) mode
    if settings_json.get_value("reconst_pipeline") is None:
        raise Exception("Photogrammetry pipeline not set")
    settings = settings_json.load()
    # Meshroom
    if int(settings["reconst_pipeline"]) == 1:
        print("start Meshroom")
        print("a folder \"Meshroom\" will be created in your project folder")
        settings["pg_cache_path"] = (path_to_str(project_paths.project_root_path / "Meshroom" / "MeshroomCache"))
        settings_json.update(settings)
        build_pg_dir(settings_json, "Meshroom")
        #run_photogrammetry_pipeline(keyframer number, project path, cache path, meshroom path, template path)
        subprocess.run(
             [
                pointmark_paths.run_photogrammetry_path,
                settings["reconst_pipeline"],
                settings["project_path"],
                settings["pg_cache_path"],
                settings["meshroom_path"],
                settings["template_path"]
            ],
            stderr = subprocess.DEVNULL,
            shell=True,
            check=True,
        )

    #Colmap
    if int(settings["reconst_pipeline"]) == 2:
        print("start Colmap")
        print("a folder \"Colmap\" will be created in your project folder")
        settings["pg_cache_path"] = path_to_str(project_paths.project_root_path / "Colmap")
        settings_json.update(settings)
        build_pg_dir(settings_json, "Colmap")
        # run_photogrammetry_pipeline(keyframer number, project path, cache path, colmap path)
        subprocess.run(
            [
                pointmark_paths.run_photogrammetry_path,
                settings["reconst_pipeline"],
                settings["project_path"],
                settings["pg_cache_path"],
                settings["colmap_path"]
            ],
            stderr=subprocess.DEVNULL,
            shell=True,
            check=True
        )

#------------------ run meshroom sfm extraction ---------------------
#runnig the added ConvertSfMFormat Node on an already built project to get sfm.json file
def run_meshroom_sfm_extraction(mod_settings_json: JsonManager, pointmark_paths: PointmarkPaths):
    print("start meshroom sfm extraction")
    mod_settings = mod_settings_json.load()
    cache_path = path_to_str(pointmark_paths.cache_path)
    mod_settings["pg_cache_path"] = path_to_str(Path(mod_settings["project_path"]) / "Meshroom" / "MeshroomCache")
    #meshroomSfmExtraction(project path, project name, meshroom path, cache path)
    subprocess.run(
        [
            pointmark_paths.meshroom_sfm_extraction,
            mod_settings["project_path"],
            mod_settings["project_name"],
            mod_settings["pg_cache_path"],
            mod_settings["meshroom_path"],
            cache_path
        ],
        stderr=subprocess.DEVNULL,
        shell=True,
        check=True
    )
