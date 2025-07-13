import h5py
import numpy as np

from src.general.python.filePaths import ProjectPaths, PointmarkPaths
from src.ReconstructionData.Adapter.MeshroomAdapter import build_sfm_json, build_reconstruction_data
from src.utils.JsonManager import JsonManager

class ReconstructionData:
    """
    Handles photogrammetric reconstruction data and provides mechanisms to
    store, retrieve, and manipulate the related datasets.

    The main purpose of this class is to manage data such as point clouds,
    camera positions, intrinsic matrices, and observations generated through
    a reconstruction pipeline. It supports saving and restoring this
    information to/from a storage file and facilitates the use of a photogrammetry
    pipeline for data generation.

    This class is designed to work with one dataset at a time. If the data
    already exists, it will restore it from storage. Otherwise, it initializes
    the reconstruction process based on the provided pipeline configuration.
    """
    def __init__(self,pointmark_paths: PointmarkPaths, project_paths: ProjectPaths, silent_print=False):
        """
        Initialize the object with given JSON settings, pointmark paths, and project paths. This constructor
        handles both the restoration of pre-saved data and the initialization of the photogrammetry pipeline
        based on the provided settings. It supports MeshroomAdapter for 3D reconstruction and explicitly
        raises an exception for unsupported or invalid configurations.

        :param settings_json: JSON manager to load configuration settings.
        :type settings_json: JsonManager

        :param pointmark_paths: Paths to pointmark resources.
        :type pointmark_paths: PointmarkPaths

        :param project_paths: Paths relevant to the project, including data and reconstruction paths.
        :type project_paths: ProjectPaths

        :raises NotImplementedError: Raised when ColmapAdapter is selected, which is currently not supported.
        :raises Exception: Raised when an unsupported photogrammetry pipeline option is provided.
        """
        settings_json = JsonManager(project_paths.settings_path, "settings.json")
        settings = settings_json.load()

        self._project_paths = project_paths
        self._store_path_pm = self._project_paths.data_path / "pointmark.h5"
        self._store_path_ci = self._project_paths.data_path / "cam_img_path.json"

        if self._has_save_data():
            self.restore(silent_print)
            return
        else:
            if not silent_print:
                print("no saved data found. in: ", self._store_path_pm)

        #start MeshroomAdapter
        if int(settings["reconst_pipeline"]) == 1:
            sfm_data_path = project_paths.sfm_data_path / "Meshroom"
            sfm_num = build_sfm_json(settings_json, pointmark_paths, project_paths)
            data = []
            for i in range(1, sfm_num + 1):
                data.append(build_reconstruction_data(sfm_data_path, i))
        # start ColmapAdapter (not Supported)
        elif int(settings["reconst_pipeline"]) == 2:
            raise NotImplementedError("Colmap not supported")
        else:
            raise Exception("Photogrammetry pipeline not existing")

        # _point_cloud:             <np.float32, 3>    pointID: <POS: X, Y, Z>
        # _camera_pos:              <np.float32, 12>   cameraID: <3x3 rotation matrix; pos: x, y, z>
        # _intrinsics:              <np.float32, 10>   intrinsicID: <focal_length(mm); distortion: k1, k2, k3, sensor_width(mm), sensor_height(mm), width(px), height(px), ppx(px), ppy(px)>
        # _camera_intrinsic_map:    <np.int32, 1>      cameraID: <intrinsicID>
        # _observations:            <np.int32, 2>      observationID: <cameraID, featureID>
        # _observation_offset:      <np.int32, 1>      pointID: <(first) observationID>
        #                                              last can be calculated by: _observation_offset[pointID+1]-1
        # _features:                <np.float32, 2>    featureID: <pos: x, y>

        #currently only one data set at a time is supported
        if not silent_print:
            print("data computed")
        self._point_cloud = data[0][0].reshape(-1, 3)
        self._camera_pos = data[0][1].reshape(-1, 12)
        self._intrinsics = data[0][2].reshape(-1, 10)
        self._camera_intrinsic_map = data[0][3].reshape(-1)
        self._observations = data[0][4].reshape(-1, 2)
        self._observation_offset = data[0][5].reshape(-1)
        self._features = data[0][6].reshape(-1, 2)
        self._cam_img_path = data[0][7]
        self.save(silent_print)

    def _has_save_data(self):
        if self._store_path_pm.exists():
            if self._store_path_ci.exists():
                return True
            else:
                raise Exception("cam_img_path.json not found, data seems corrupted")
        elif self._store_path_ci.exists():
            raise Exception("pointmark.h5 not found, data seems corrupted")
        else:
            return False

    def save(self, silent_print):
        """
        Saves the current attributes of the object as datasets within an HDF5 file.
        The file is created at the path specified in the `_store_path` attribute,
        and each dataset is compressed using the gzip compression method.

        The function serializes the following attributes into the HDF5 file:
        - `_point_cloud`
        - `_camera_pos`
        - `_intrinsics`
        - `_camera_intrinsic_map`
        - `_observations`
        - `_observation_offset`
        - `_features`

        :return: None
        """
        if not silent_print:
            print("saving data at: ", self._store_path_pm)
        data = {
            "point_cloud": self._point_cloud,
            "camera_pos": self._camera_pos,
            "intrinsics": self._intrinsics,
            "camera_intrinsic_map": self._camera_intrinsic_map,
            "observations": self._observations,
            "observation_offset": self._observation_offset,
            "features": self._features,
        }

        with h5py.File(self._store_path_pm, "w") as file:
            for name, data_array in data.items():
                file.create_dataset(name, data=data_array, compression="gzip")

        if not silent_print:
            print("saving data at: ", self._store_path_ci)
        cam_img_path_json = JsonManager(self._store_path_ci.parent, self._store_path_ci.name)
        cam_img_path_json.update(self._cam_img_path)

    def restore(self, silent_print=False):
        """
        Restores data from the specified storage path, loading various elements from
        an HDF5 file such as point clouds, camera positions, intrinsics, and other
        observational and feature-related data. This method is designed to populate
        the corresponding attributes of the object with the data extracted from the
        file.

        :raises FileNotFoundError: If the file at the specified store path does not exist.
        :raises OSError: If the file cannot be opened or read for any reason.
        """
        if not silent_print:
            print("restoring data from: ", self._store_path_pm)
        with h5py.File(self._store_path_pm, 'r') as file:
            self._point_cloud = file["point_cloud"][:]
            self._camera_pos = file["camera_pos"][:]
            self._intrinsics = file["intrinsics"][:]
            self._camera_intrinsic_map = file["camera_intrinsic_map"][:]
            self._observations = file["observations"][:]
            self._observation_offset = file["observation_offset"][:]
            self._features = file["features"][:]

        if not silent_print:
            print("restoring data from: ", self._store_path_ci)
        cam_img_path_json = JsonManager(self._store_path_ci.parent, self._store_path_ci.name)
        self._cam_img_path = cam_img_path_json.load()

    def set_point_cloud(self, point_cloud: np.ndarray) -> None:
        self._point_cloud = point_cloud

    def get_point_cloud(self) -> np.ndarray:
        return self._point_cloud

    def set_camera_pos(self, camera_pos: np.ndarray) -> None:
        self._camera_pos = camera_pos

    def get_camera_poses(self) -> np.ndarray:
        return self._camera_pos

    def set_camera_intrinsic_map(self, camera_intrinsic_map: np.ndarray) -> None:
        self._camera_intrinsic_map = camera_intrinsic_map

    def get_camera_intrinsic_map(self) -> np.ndarray:
        return self._camera_intrinsic_map

    def set_observations(self, observations: np.ndarray) -> None:
        self._observations = observations

    def get_observations(self) -> np.ndarray:
        return self._observations

    def get_observation_offset(self) -> np.ndarray:
        return self._observation_offset

    def set_intrinsics(self, intrinsics: np.ndarray) -> None:
        self._intrinsics = intrinsics

    def get_intrinsics(self) -> np.ndarray:
        return self._intrinsics

    def set_features(self, features: np.ndarray) -> None:
        self._features = features

    def get_features(self) -> np.ndarray[np.float32, np.float32]:
        return self._features

    def set_cam_img_path(self, cam_img_path: dict) -> None:
        self._cam_img_path = cam_img_path

    def get_cam_img_path(self) -> dict[str, str]:
        return self._cam_img_path

