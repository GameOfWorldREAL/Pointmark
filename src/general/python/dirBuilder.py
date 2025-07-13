from pathlib import Path
from src.general.python.filePaths import ProjectPaths, BatchPaths
from src.utils.JsonManager import JsonManager

#building directories

#build the photogrammetry directories inside a project
def build_pg_dir(settings_json: JsonManager, pg_pipeline_name: str):
    project_paths = ProjectPaths(Path(settings_json.get_value("project_path")))
    (project_paths.sfm_data_path / pg_pipeline_name).mkdir(parents=True, exist_ok=True)
    (project_paths.project_root_path / pg_pipeline_name).mkdir(parents=True, exist_ok=True)
    Path(settings_json.get_value("pg_cache_path")).mkdir(parents=True, exist_ok=True)

#build the pointmark directories inside a project
def build_pointmark_dir(project_paths: ProjectPaths):
    project_paths.pointmark_path.mkdir(parents=True, exist_ok=True)
    project_paths.settings_path.mkdir(parents=True, exist_ok=True)
    project_paths.sfm_data_path.mkdir(parents=True, exist_ok=True)
    project_paths.data_path.mkdir(parents=True, exist_ok=True)
    project_paths.images_path.mkdir(parents=True, exist_ok=True)

#build the batch directories inside a project
def build_batch_dir(project_paths: BatchPaths):
    project_paths.settings_path.mkdir(parents=True, exist_ok=True)
    project_paths.projects_path.mkdir(parents=True, exist_ok=True)

#build the batch (videos) directories inside a project
def build_batch_videos_dir(project_paths: BatchPaths):
    build_batch_dir(project_paths)
    project_paths.videos_path.mkdir(parents=True, exist_ok=True)
