import argparse
import sys
from pathlib import Path

pointmark_path = Path(__file__).resolve().parent
src_path = str(pointmark_path / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.pointmark.setup.batch_setup import init_batch
from src.pointmark.batchRun.batchFromImages import batchFromImage
from src.pointmark.batchRun.batchFromPG import batchFromPG
from src.pointmark.batchRun.batchFromVideo import batchFromVideo
from src.pointmark.setup.setup import print_pointmark_logo, setup_UI

# Meshroom:
#   if --sfm_data_path set, folder containing a single sfm.json in another folder with project name, any .json will be computed!.
#   and any folder will cause the program to try building a project out of it.
#   Images should be in the same space as the meshroom pipeline run on it or else the images can't be found.
#   Additionally, all images expected to be in the same folder
#   only one sfm.json file is supported per project
# Colmap:
#   not implemented yet

def main():
    parser = argparse.ArgumentParser(description="batch Pointmark runner")
    parser.add_argument(
        "-p", "--project_path",
        help="the folder the project data will be stored",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-s", "--setup_path",
        help="path containing the setup",
        type=Path
    )
    parser.add_argument(
        "-m", "--metrics",
        action="store_true",
        help="compute metrics and store them in the project folder"
    )
    parser.add_argument(
        "--ooi",
        action="store_true",
        help="compute object of interest"
    )
    parser.add_argument(
        "--aggregation_mode",
        choices=["min", "max", "mean", "none"],
        default="none",
        help="aggregation mode: 'min', 'max', 'mean' or 'none'"
    )
    parser.add_argument(
        "-e", "--export",
        action="store_true",
        help="export the results to a json file"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--image_sets_path",
        nargs="?", const=None, default=argparse.SUPPRESS,
        type=Path,
        help="the folder containing the image sets (optional: ohne Pfad → None)"
    )
    group.add_argument(
        "-d", "--sfm_data_path",
        nargs="?", const=None, default=argparse.SUPPRESS,
        type=Path,
        help="the folder containing the sfm data sets (optional: ohne Pfad → None)"
    )
    group.add_argument(
        "-v", "--video_files_path",
        nargs="?", const=None, default=argparse.SUPPRESS,
        type=Path,
        help="the folder containing the video files (optional: ohne Pfad → None)"
    )
    args = parser.parse_args()

    print_pointmark_logo()
    setup_UI()

    print("pointmark batch pipeline")

    if hasattr(args, "video_files_path"):
        warning = "warning: video files expected to be in: project_path/videos"
        init_batch(
            args,
            args.video_files_path,
            "batch video",
            warning,
            batchFromVideo)

    elif hasattr(args, "image_sets_path"):
        warning = "warning: expected images in ready Pointmark_batch project structure inside the project folder"
        init_batch(
            args,
            args.image_sets_path,
            "batch images",
            warning,
            batchFromImage)

    else:  # sfm_data_path
        warning = "warning: expected sfm_data in ready Pointmark_batch project structure inside the project folder"
        init_batch(
            args,
            args.sfm_data_path,
            "batch photogrammetry data",
            warning,
            batchFromPG)

if __name__ == "__main__":
    main()
