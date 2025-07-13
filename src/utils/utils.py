import os
import shutil
from pathlib import Path

image_ext = [".jpg", ".jpeg", ".png", ".webp"]
video_ext = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]

#ask user to press "enter" to exit
def ask_to_exit(skip=False):
    if not skip:
        choice = input("do you want to exit press Y else press N:\n")
        if choice.upper() == "Y":
            exit(0)
    else:
        exit(0)

#ask user to press "enter" to continue
def enter_to_continue(skip=False):
    if not skip:
        input("press enter to continue...")

#copy all videos in a folder to another folder
def copy_videos(source: Path, destination: Path):
    for file in source.iterdir():
        if file.suffix.lower() in video_ext:
            shutil.copy(file, destination)

#copy all images in a folder to another folder
def copy_images(source: Path, destination: Path):
    for file in source.iterdir():
        if file.is_file() and file.suffix.lower() in image_ext:
            shutil.copy2(str(file), str(destination))

#detect all video files in a folder and return their paths as a list
def detect_videos(video_path: Path, print_out=False):
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
    videos = []
    for file in video_path.iterdir():
        if file.suffix.lower() in video_extensions:
            videos.append(path_to_str(file))
            if print_out:
                print(file.name)
    return videos

#return the number of SfM files in a folder
def num_sfm_file(folder: Path) -> int:
    return sum(1 for file in folder.iterdir() if file.is_file() and file.suffix.lower() == '.json')

#check if a string is an integer
def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def path_to_str(path: Path | str) -> str:
    """
    Convert a provided file system path to a string representation that is compatible
    with the operating system. For `posix` systems such as Linux or macOS, it will
    return the path in POSIX format. For `nt` systems like Windows, it will convert
    forward slashes in the path to backward slashes. If the operating system is not
    supported, an exception is raised.

    :param path: A file system path. It can be provided as a string or a Path object.
    :returns: A string representation of the path compatible with the operating system.
    :raises Exception: If the operating system is not supported.
    """
    path = Path(path)
    #linux/macos
    if os.name == "posix":
        return path.as_posix()
    #windows
    elif os.name == "nt":
        return str(path).replace("/", "\\")
    else:
        raise Exception(f"operating system not supported: {os.name}")