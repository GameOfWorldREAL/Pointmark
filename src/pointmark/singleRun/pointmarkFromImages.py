import sys
from pathlib import Path
pointmark_path = Path(__file__).resolve().parent
src_path = str(pointmark_path / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.general.python.cmdRunner import run_photogrammetry
from src.general.python.dirBuilder import  build_pointmark_dir
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
from src.pointmark.setup.setup import setup_single_pipeline, prepare_project
from src.pointmark.singleRun.pointmarkFromPG import pointmarkFromPG
from src.utils.JsonManager import JsonManager
from src.utils.setup_utils import print_settings_found
from src.utils.utils import copy_images

def pointmarkFromImages(pointmark_paths: PointmarkPaths, project_path: Path, images_path: Path = None, skip=False,
                        metrics=False, aggregation_mode="min", ooi=True, export: Path = None):
    print(project_path)
    project_paths = ProjectPaths(project_path)
    build_pointmark_dir(project_paths)
    settings_json = JsonManager(project_paths.settings_path, "settings.json")
    if not images_path is None:
        copy_images(images_path, project_paths.images_path)
        print_settings_found(settings_json)
        setup_single_pipeline(settings_json, pointmark_paths, project_paths, skip_video=True, skip=skip)
        prepare_project(settings_json, project_paths)
    run_photogrammetry(settings_json, pointmark_paths, project_paths)
    print("-----------------------------------------")

    pointmarkFromPG(pointmark_paths, project_paths.project_root_path, skip=skip, metrics=metrics,
                    aggregation_mode=aggregation_mode, ooi=ooi, export=export)