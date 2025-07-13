import multiprocessing
import shutil
from pathlib import Path
import numpy as np

from src.general.python.cmdRunner import run_meshroom_sfm_extraction
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
from src.utils.JsonManager import JsonManager
from src.utils.utils import num_sfm_file, path_to_str


def build_sfm_json(settings_json: JsonManager, pointmark_paths: PointmarkPaths, project_paths: ProjectPaths):
    print("  building SfM data")
    n_sfm = num_sfm_file(project_paths.sfm_data_path / "Meshroom")
    if n_sfm > 0: #modify for multiple sfm support later
        print("  SfM data already built")
        return n_sfm

    settings = settings_json.load()
    mroom_project = project_paths.project_root_path / "Meshroom"
    convert_sfm_path = pointmark_paths.templates_path
    cache_path = pointmark_paths.cache_path

    mg_files = list(mroom_project.glob("*.mg"))
    # check if only a single .mg file exists else raise an exception
    if len(mg_files) > 1:
        raise Exception("Multiple .mg files found in project: " + path_to_str(mroom_project))
    if len(mg_files) == 0:
        raise Exception("No .mg files found in project: " + path_to_str(mroom_project))

    mroom_mg = mg_files[0]
    shutil.copy(mroom_mg, cache_path)

    # continue
    mroom_file_json = JsonManager(cache_path, mg_files[0].name)
    mroom_file = mroom_file_json.load()
    sfm_nodes = []
    convert_counter = 1
    for name, node in mroom_file["graph"].items():
        if node["nodeType"] == "StructureFromMotion":
            sfm_nodes.append(
                (name, "5d7bff09a646a550fedbbf67efeab91e9b02218f"))  # generate an ID for each convertSfMFormat node
        elif node["nodeType"] == "ConvertSfMFormat":  # count existing ConvertSfMFormat
            convert_counter += 1

    if len(sfm_nodes) == 0:
        raise Exception("no structure from motion nodes found in project: " + path_to_str(mroom_project))
    if len(sfm_nodes) > 1:
        raise NotImplementedError(
            "currently more than one structure from motion nodes not supported: " + path_to_str(mroom_project))

    for sfm_node in sfm_nodes:  # build as loop for later expansion
        #check if a file already exists:
        sfm_json_path = project_paths.sfm_data_path / "Meshroom" / ("sfm_" + str(convert_counter) + ".json")
        if sfm_json_path.exists():
            print(("sfm_" + str(convert_counter) + ".json") + " already exists in: " + path_to_str(mroom_project))
            (cache_path / mroom_mg.name).unlink()
            return len(sfm_nodes)

        # setup pipeline
        convert_sfm_json = JsonManager(convert_sfm_path, "convertSfMFormat_temp.mg")
        convert_sfm = convert_sfm_json.load()
        convert_sfm["inputs"]["input"] = "{" + sfm_node[0] + ".output}"
        convert_sfm["inputs"]["describerTypes"] = "{" + sfm_node[0] + ".describerTypes}"
        mroom_file["graph"]["ConvertSfMFormat_" + str(convert_counter)] = convert_sfm
        mroom_file["header"]["nodesVersions"]["ConvertSfMFormat"] = "2.0"
        mroom_file_json.update(mroom_file)

        #store folders already build in ConvertSfMFormat
        ConvertSfMFormat_path = Path(settings["pg_cache_path"]) / "ConvertSfMFormat"
        if ConvertSfMFormat_path.exists():
            old_folders = {f for f in ConvertSfMFormat_path.iterdir() if f.is_dir()}
        else:
            old_folders = set()
        # set up Meshroom execution:
        shutil.copy(project_paths.settings_path / "settings.json", cache_path)
        mod_settings_json = JsonManager(cache_path, "settings.json")
        mod_settings = mod_settings_json.load()
        mod_settings["project_name"] = mroom_mg.stem
        mod_settings_json.update(mod_settings)
        mod_settings_json.filepath = path_to_str(cache_path / "settings.json")
        run_meshroom_sfm_extraction(mod_settings_json, pointmark_paths)
        print("-----------------------------------------")

        # remove cache files
        (cache_path / "settings.json").unlink()
        (cache_path / mroom_mg.name).unlink()

        # move sfm.json to the correct dir
        new_folders = {f for f in ConvertSfMFormat_path.iterdir() if f.is_dir()}
        convert_sfm_folders = new_folders - old_folders

        if len(convert_sfm_folders) != 1:
            raise Exception(
                "unexpected number of sfm_json_folders: " + str(len(convert_sfm_folders)) + "in: " + path_to_str(ConvertSfMFormat_path))

        sfm_json_folder = convert_sfm_folders.pop()
        source = sfm_json_folder / "sfm.json"
        destination = project_paths.sfm_data_path / "Meshroom" / ("sfm_" + str(convert_counter) + ".json")

        shutil.move(source, destination)
        # remove newly generated dirs
        shutil.rmtree(Path(sfm_json_folder))
        if not any((Path(settings["pg_cache_path"]) / "ConvertSfMFormat").iterdir()):
            shutil.rmtree(Path(settings["pg_cache_path"]) / "ConvertSfMFormat")
        convert_counter += 1

    return len(sfm_nodes)

#--------------------- helper functions ------------------------
def searchID(map_table: dict, real_id: int):
    return map_table[real_id]

def searchView(sfm_data, real_id: int):
    for pos, view in enumerate(sfm_data["views"]):
        if int(view["viewId"]) == real_id:
            return pos
    raise KeyError(f"view id not found: /{real_id}/")


def _compute_feature_num(landmarks):
    if len(landmarks) < 20000000:
        #single core execution for normal datasets
        return sum(len(landmark["observations"]) for landmark in landmarks)
    else:
        # multicore execution for very large datasets > 20.000.000 landmarks TODO maybe remove
        num_landmarks = len(landmarks)
        landmarks = landmarks
        observations_list = np.empty(num_landmarks, dtype=object)
        for landmark in range(num_landmarks):
            observations_list[landmark] = landmarks[landmark]["observations"]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            counts = pool.map(count_features, observations_list, chunksize=10000)
        return sum(counts)

def count_features(observations):
    return len(observations)
#---------------------------------------------------------------

def build_reconstruction_data(sfm_data_path: Path, iteration):
    print("  building reconstruction data")
    sfm_json = JsonManager(sfm_data_path, "sfm_" + str(iteration) + ".json")
    sfm_data = sfm_json.load()
    #get intrinsics
    _intrinsic_id_map = {}
    intrinsics = np.empty(((len(sfm_data["intrinsics"])), 10), dtype=np.float32)
    for intrinsicId in range(len(sfm_data["intrinsics"])):
        # remap intrinsic id's for more efficient access
        _intrinsic_id_map[int(sfm_data["intrinsics"][intrinsicId]["intrinsicId"])] = intrinsicId
        intrinsic_data = sfm_data["intrinsics"][intrinsicId]

        # compute camera calibration matrix https://de.mathworks.com/help/vision/ug/camera-calibration.html
        focal_length = np.atleast_1d(np.array(intrinsic_data["focalLength"], dtype=np.float32))
        pixel_x_size = np.float32(intrinsic_data["width"])
        pixel_y_size = np.float32(intrinsic_data["height"])
        sensor_x_size = np.float32(intrinsic_data["sensorWidth"])
        sensor_y_size = np.float32(intrinsic_data["sensorHeight"])
        ppx = np.float32(intrinsic_data["principalPoint"][0])  # in mm
        ppy = np.float32(intrinsic_data["principalPoint"][1])  # in mm

        distortion = np.array(intrinsic_data["distortionParams"], dtype=np.float32)
        sensor_size = np.array([sensor_x_size, sensor_y_size], dtype=np.float32)
        resolution = np.array([pixel_x_size, pixel_y_size], dtype=np.float32)
        principal_point = np.array([ppx, ppy], dtype=np.float32)

        intrinsics[intrinsicId] = np.concatenate((focal_length, distortion, sensor_size, resolution, principal_point))

    #get camera pose
    _camera_id_map = {}
    camera_pos = np.empty((len(sfm_data["poses"]), 12), dtype=np.float32)
    for cameraId in range(len(sfm_data["poses"])):
        camera_data = sfm_data["poses"][cameraId]
        #remap camera id's for more efficient access
        _camera_id_map[int(camera_data["poseId"])] = cameraId
        #get rotation 3x3 matrix and camera pos
        rotation = np.array(camera_data["pose"]["transform"]["rotation"], dtype=np.float32)
        center = np.array(camera_data["pose"]["transform"]["center"], dtype=np.float32)
        camera_pos[cameraId] = np.concatenate((rotation, center))

    #build a camera-intrinsics map and camera-image_path directory
    camera_intrinsic_map = np.empty(len(_camera_id_map), dtype=np.int32)
    camera_image_path = {}
    for view_id in _camera_id_map:
        view = searchView(sfm_data, view_id)
        intrinsic_id = int(sfm_data["views"][view]["intrinsicId"])
        intr_id_map = searchID(_intrinsic_id_map, intrinsic_id)
        cam_id_map = searchID(_camera_id_map, view_id)
        camera_intrinsic_map[cam_id_map] = intr_id_map

        image_path = path_to_str(Path(sfm_data["views"][view]["path"]))
        camera_image_path[cam_id_map] = image_path

    # get point cloud observations and features
    landmarks = sfm_data["structure"]
    point_cloud = np.empty((len(landmarks), 3), dtype=np.float32)
    feat_num = _compute_feature_num(landmarks)
    features = np.empty((feat_num, 2), dtype=np.float32)
    observations = np.empty((feat_num, 2), dtype=np.int32)
    observation_offset = np.empty((len(landmarks) + 1), dtype=np.int32)

    feat_cntr = 0
    for landmark in range(len(landmarks)):
        landmark_data = landmarks[landmark]

        #store landmark into point cloud
        point_cloud[landmark] = np.array(landmark_data["X"], dtype=np.float32)

        #store features and assign a landmark the cameraID and featureID in observations
        #observation_offset stores the beginning of an observation list in observations
        for observation in landmark_data["observations"]:
            features[feat_cntr] = np.array(observation["x"], dtype=np.float32)
            observations[feat_cntr] = (searchID(_camera_id_map, int(observation["observationId"])), feat_cntr)
            feat_cntr = feat_cntr + 1
        observation_offset[landmark] = feat_cntr-len(landmark_data["observations"])
    observation_offset[len(landmarks)] = feat_cntr #set last value

    return point_cloud, camera_pos, intrinsics, camera_intrinsic_map, observations, observation_offset, features, camera_image_path

