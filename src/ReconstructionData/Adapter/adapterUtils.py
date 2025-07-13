import shutil
from pathlib import Path

from src.general.python.filePaths import ProjectPaths
from src.utils.JsonManager import JsonManager
from src.utils.utils import num_sfm_file, path_to_str


#  -------------------------- runFromPG functions -----------------------------------
# Interface for sfm data copy
def copy_sfm_data(pipeline: int, src_sfm_path: Path, project_paths: ProjectPaths):
    print(src_sfm_path)
    if pipeline == 1:
        _meshroom_copy_sfm_data(pipeline, src_sfm_path, project_paths)
    if pipeline == 2:
        _colmap_copy_sfm_data(pipeline, src_sfm_path, project_paths)

# returns the path to the sfm data folder for the given pipeline
def get_sfm_path(pipeline: int, project_paths:ProjectPaths):
    if pipeline == 1:
        path = project_paths.sfm_data_path / "Meshroom"
    elif pipeline == 2:
        path = project_paths.sfm_data_path / "Colmap"
    else:
        raise Exception("invalid pipeline")

    path.mkdir(parents=True, exist_ok=True)
    return path

#----------------------------------- Meshroom utils -------------------------------------
#copy images and sfm data to the project folder
def _meshroom_copy_sfm_data(pipeline: int, src_sfm_path: Path, project_paths: ProjectPaths):
    sfm_data_path = get_sfm_path(pipeline, project_paths)
    sfm_data_path.mkdir(parents=True, exist_ok=True)
    #copy sfm data
    sfm_count = 1                                           #modify for multiple sfm support
    sfm_json_name = "sfm_" + str(sfm_count) + ".json"
    src_sfm_json = Path(get_sfm_json_path(src_sfm_path))
    if num_sfm_file(sfm_data_path) > 0:
        print("  SfM file already exists, skipping copy")
    else:
        print("  copying sfm data")
        shutil.copy(src_sfm_json, sfm_data_path / sfm_json_name)
    #copy images and modify images paths in data to match the new location
    if any(project_paths.images_path.iterdir()):
        print("  images already exists, skipping copy")
    else:
        print("  copying images")
        sfm_json = JsonManager(sfm_data_path, sfm_json_name)
        sfm = sfm_json.load()
        for view in sfm["views"]:
            old_image_path = Path(view["path"])
            shutil.copy2(old_image_path, project_paths.images_path)
            view["path"] = str(project_paths.images_path / old_image_path.name)
        sfm_json.update(sfm)

#searches and returns the first sfm.json file in the sfm_data_dir
def get_sfm_json_path(sfm_data_dir):
    for sfm in sfm_data_dir.iterdir():
        if sfm.suffix.lower() == ".json":
            return path_to_str(sfm)
    return None

#----------------------------------- Colmap utils -------------------------------------
#not supported yet
def _colmap_copy_sfm_data(pipeline: int, src_sfm_path: Path, project_paths: ProjectPaths):
    raise NotImplementedError()