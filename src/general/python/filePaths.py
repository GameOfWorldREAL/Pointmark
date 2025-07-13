from pathlib import Path

# path structure for Pointmark
class PointmarkPaths:
    """
    Manages paths required for Pointmark operations.

    This class encapsulates paths to various directories and command files used for
    different functionalities in the Pointmark application. It initializes the paths
    based on a root directory provided at initialization. These paths are essential
    for managing caching, templates, keyframe selection, and executing certain scripts.

    :ivar pointmark_root_path: The root directory for Pointmark operations.
    :type pointmark_root_path: Path

    :ivar cache_path: Directory path for caching Pointmark data.
    :type cache_path: Path

    :ivar templates_path: Path to the templates resource directory.
    :type templates_path: Path

    :ivar keyframe_selector_path: Path to the keyframe selection resource directory.
    :type keyframe_selector_path: Path

    :ivar keyframe_selection_cmd: Path to the script for keyframe selection.
    :type keyframe_selection_cmd: Path

    :ivar run_photogrammetry_path: Path to the script for executing photogrammetry.
    :type run_photogrammetry_path: Path

    :ivar meshroom_sfm_extraction: Path to the script for extracting SFM using Meshroom.
    :type meshroom_sfm_extraction: Path
    """
    def __init__(self, pointmark_path: Path):
        #third party and resources
        self.pointmark_root_path = pointmark_path
        self.cache_path = pointmark_path / "cache"
        self.templates_path = pointmark_path / "resources" / "templates"
        self.keyframe_selector_path = pointmark_path / "keyframeSelection"

        #code related
        self.keyframe_selection_cmd = pointmark_path / "src" / "general" / "cmd" / "keyframeSelection.cmd"
        self.run_photogrammetry_path = pointmark_path / "src" / "general" / "cmd" / "runPhotogrammetry.cmd"
        self.meshroom_sfm_extraction = pointmark_path / "src" / "general" / "cmd" / "meshroomSfmExtraction.cmd"

# path structure for a project
class ProjectPaths:
    """
    Handles and provides standardized paths for project-related directories.

    This class is intended to centralize the management of paths within a project
    structure. With a given root project path, it constructs paths to key directories
    and provides easy access to these locations.

    Users can utilize this class to ensure consistent directory management and to avoid
    hardcoding paths throughout the project. It is especially helpful for projects that
    require specific subdirectory structures.

    :ivar project_root_path: The root directory path of the project.
    :type project_root_path: Path

    :ivar images_path: The path to the directory containing image files.
    :type images_path: Path

    :ivar pointmark_path: The path to the Pointmark directory specific to the project.
    :type pointmark_path: Path

    :ivar sfm_data_path: The path to the sfmData subdirectory within Pointmark.
    :type sfm_data_path: Path

    :ivar settings_path: The path to the settings subdirectory within Pointmark.
    :type settings_path: Path

    :ivar data_path: The path to the data subdirectory within Pointmark.
    :type data_path: Path
    """
    def __init__(self, project_path: Path):
        self.project_root_path = project_path
        self.images_path = self.project_root_path / "images"
        self.pointmark_path = project_path / "Pointmark"
        self.sfm_data_path = self.pointmark_path / "sfmData"
        self.settings_path = self.pointmark_path / "settings"
        self.data_path = self.pointmark_path / "data"
        self.export_path = self.project_root_path / "export"

# path structure for a batch project
class BatchPaths:
    """
    Represents a structured path configuration used for managing project-related
    directory paths within a specific project root.

    This class facilitates the definition and organization of paths related to videos,
    projects, settings, and data directories, relative to a given project root.

    :ivar project_root_path: The root directory of the project.
    :type project_root_path: Path

    :ivar videos_path: The directory path where video files are stored, relative to the project root.
    :type videos_path: Path

    :ivar projects_path: The directory path where project-related files are stored, relative to the project root.
    :type projects_path: Path

    :ivar settings_path: The directory path where settings-related files are stored, relative to the project root.
    :type settings_path: Path

    :ivar data_path: The directory path for accessing data, which corresponds to the settings directory.
    :type data_path: Path
    """
    def __init__(self, project_path: Path):
        self.project_root_path = project_path
        self.videos_path = self.project_root_path / "videos"
        self.projects_path = self.project_root_path / "projects"
        self.settings_path = self.project_root_path / "settings"
        self.data_path = self.settings_path