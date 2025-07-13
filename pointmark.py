import argparse
import sys
from pathlib import Path

pointmark_path = Path(__file__).resolve().parent
src_path = str(pointmark_path / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.pointmark.setup.setup import init_pointmark, setup_UI, print_pointmark_logo
from src.pointmark.singleRun.pointmarkFromImages import pointmarkFromImages
from src.pointmark.singleRun.pointmarkFromPG import pointmarkFromPG
from src.pointmark.singleRun.pointmarkFromVideo import pointmarkFromVideo



def main():
    _FLAG_ONLY = object()   #identify if a flag is set without providing a path argument

    parser = argparse.ArgumentParser(description="batch Pointmark runner")
    parser.add_argument(
        "-p", "--project_path",
        help="the folder the project data will be stored",
        type=Path,
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="all questions will be answered with yes"
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
        default=None,
        help="aggregation mode: 'min', 'max', 'mean' or 'none'"
    )
    parser.add_argument(
        "-e", "--export",
        nargs="?",
        default=None,
        const=_FLAG_ONLY,
        type=Path,
        help="export the results to a json file"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--images_path",
        nargs="?",
        const=None,
        default=argparse.SUPPRESS,
        type=Path,
        help="the folder containing the images (optional: if no path follows, will pass None)"
    )
    group.add_argument(
        "-d", "--sfm_data_path",
        nargs="?",
        const=None,
        default=argparse.SUPPRESS,
        type=Path,
        help="the folder containing the sfm data (optional: if no path follows, will pass None)"
    )
    group.add_argument(
        "-v", "--video_path",
        nargs="?",
        const=None,
        default=argparse.SUPPRESS,
        type=Path,
        help="the folder containing the video files (optional: if no path follows, will pass None)"
    )
    args = parser.parse_args()

    setup_UI()
    print_pointmark_logo()
    print("pointmark single pipeline")

    # select correct pipeline
    if hasattr(args, "video_path"):
        warning = "warning: video file expected to be in: project_path/video"
        init_pointmark(
            args,
            args.video_path,
            "video",
            warning,
            pointmarkFromVideo,
            _FLAG_ONLY)

    elif hasattr(args, "images_path"):
        warning = "warning: expected images in ready Pointmark project structure inside images folder"
        init_pointmark(
            args,
            args.images_path,
            "images",
            warning,
            pointmarkFromImages,
            _FLAG_ONLY)

    else:  # sfm_data_path
        warning = "warning: expected sfm data in ready Pointmark project structure inside the sfm_data folder"
        init_pointmark(
            args,
            args.sfm_data_path,
            "photogrammetry data",
            warning,
            pointmarkFromPG,
            _FLAG_ONLY)

if __name__ == "__main__":
    main()