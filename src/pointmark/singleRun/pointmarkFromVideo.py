
import sys
from pathlib import Path
pointmark_path = Path(__file__).resolve().parent
src_path = str(pointmark_path / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import shutil

from src.general.python.cmdRunner import run_keyframe_selection
from src.general.python.dirBuilder import build_pointmark_dir
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
from src.pointmark.setup.setup import setup_single_pipeline, prepare_project
from src.pointmark.singleRun.pointmarkFromImages import pointmarkFromImages
from src.utils.JsonManager import JsonManager
from src.utils.setup_utils import print_settings_found


def _keyframe_extraction(pointmark_paths: PointmarkPaths, project_paths: ProjectPaths):
    project_path = project_paths.project_root_path
    print("processing images for project:", project_path)
    print("processing images for project:", project_path)
    project_paths = ProjectPaths(project_path)
    settings_json = JsonManager(project_paths.settings_path, "settings.json")
    run_keyframe_selection(settings_json, pointmark_paths, project_paths)

def pointmarkFromVideo(pointmark_paths: PointmarkPaths, project_path: Path = None, src_video_path: Path = None, skip=False,
                       metrics=False, aggregation_mode="min", ooi=True, export: Path = None,):
    print(project_path)
    project_paths = ProjectPaths(project_path)
    build_pointmark_dir(project_paths)
    settings_json = JsonManager(project_paths.settings_path, "settings.json")

    dest_video_path = project_paths.project_root_path / "video"
    if not src_video_path is None:
        if not src_video_path is dest_video_path:
            dest_video_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video_path, dest_video_path)
        print_settings_found(settings_json)
        video_path = dest_video_path / src_video_path.name
        setup_single_pipeline(settings_json, pointmark_paths, project_paths, video_path, skip_video=False, skip=skip)
        prepare_project(settings_json, project_paths, video_path)
    run_keyframe_selection(settings_json, pointmark_paths, project_paths)
    pointmarkFromImages(pointmark_paths, project_paths.project_root_path, skip=skip, metrics=metrics,
                        aggregation_mode=aggregation_mode, ooi=ooi, export=export)