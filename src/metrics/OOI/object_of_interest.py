# Imports
#setup, third party, local TODO make single setup file
import sys
from pathlib import Path

from src.utils.utils import path_to_str
pointmark_path = Path(__file__).resolve().parents[4]
sys.path.append(path_to_str(pointmark_path / "third" / "Pointnet_Pointnet2_pytorch"))

from scipy.spatial import cKDTree
from src.metrics.metrics_norm_aggr import compute_mean_var, L
import warnings
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from src.ReconstructionData.ReconstructionData import ReconstructionData
from src.general.python.filePaths import PointmarkPaths, ProjectPaths
#from third.PointTransformerV3.model import PointTransformerV3 TODO

# from third.BiRefNet.models.birefnet import BiRefNet
# from third.BiRefNet.utils import check_state_dict
#
# birefnet = BiRefNet(bb_pretrained=False)
# state_dict = torch.load(PATH_TO_WEIGHT, map_location='cpu')
# state_dict = check_state_dict(state_dict)
# birefnet.load_state_dict(state_dict)
#
# torch.set_float32_matmul_precision(['high', 'highest'][0])
# birefnet.to('cuda')
# birefnet.eval()
# birefnet.half()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESERVED_BYTES = 2*1024

def compute_object_of_interest(pointmark_paths: PointmarkPaths, project_paths: ProjectPaths, density_metric: np.ndarray):
    # ------------------------
    # https://huggingface.co/ZhengPeng7/BiRefNet
    # Ref: Zheng et al. (2024) – BiRefNet segmentation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
        if device.type == "cuda":
            birefnet = birefnet.to(device).half()
        else:
            birefnet = birefnet.to(device)
        birefnet.eval()
    #------------------------
    recon_data = ReconstructionData(pointmark_paths, project_paths, silent_print=True)
    camera_img_path = recon_data.get_cam_img_path()
    camera_num = recon_data.get_camera_poses().shape[0]
    image_paths = []
    for cam in range(camera_num):
        image_paths.append(camera_img_path[str(cam)])
    segmentation_mask = extract_objects(birefnet, image_paths)

    point_cloud = recon_data.get_point_cloud()

    point_rating = np.zeros(point_cloud.shape[0])
    observations = recon_data.get_observations()
    observation_offset = recon_data.get_observation_offset()
    features = recon_data.get_features()
    for point_id in tqdm(range(point_rating.shape[0]), desc="  build point rating"):
        observation = observations[observation_offset[point_id]:observation_offset[point_id+1]]
        for cam_id, feature_id in observation:
            feature = features[feature_id]
            x = int(round(feature[0]))
            y = int(round(feature[1]))
            point_rating[point_id] += segmentation_mask[cam_id][y, x]
    mean, var = compute_mean_var(point_rating)
    norm_point_rating = L(point_rating, mean, var)
    mask = norm_point_rating > 0.4

    #filter by density
    knn = 20
    kdtree = cKDTree(point_cloud)
    _, neighbors = kdtree.query(point_cloud[mask], k=knn)
    ooi_neighbors = mask[neighbors.reshape(-1)]
    ooi_neighbors = ooi_neighbors.reshape(-1, knn)
    ooi_percent = ooi_neighbors.sum(axis=1) / knn

    mean, var = compute_mean_var(density_metric)
    norm_density = L(density_metric, mean, var)

    ooi_density = norm_density[mask]
    ooi = ooi_percent / ooi_density
    mask_select = ooi > 0.4

    filtered_mask = np.zeros(point_cloud.shape[0], dtype=bool)
    inital_ids = np.where(mask)[0]
    selected_ids = inital_ids[mask_select]
    filtered_mask[selected_ids] = True

    #refine point rating
    radius = compute_radius(density_metric, 4)
    ooi_points = point_cloud[filtered_mask]

    neighbors = list(kdtree.query_ball_point(ooi_points, r=radius))
    refined_mask = filtered_mask.copy()
    for i in tqdm(range(len(neighbors)), desc="  refine point rating"):
        neighbor_list =  np.array(neighbors[i])

        mask_over_neighbors = filtered_mask[neighbor_list]
        ooi_ids = neighbor_list[mask_over_neighbors]
        neighbor_ooi = point_cloud[ooi_ids]
        if neighbor_ooi.shape[0] == 0:
            continue

        dir_vector = np.sum(neighbor_ooi - ooi_points[i], axis=0) / neighbor_ooi.shape[0]

        norm = np.sqrt(np.sum(dir_vector**2))

        if norm > 1e-12:
            dir_norm = dir_vector / norm
        else:
            dir_vector = np.zeros(3)
            dir_norm = np.array([1,0,0])

        plane_vector = -dir_norm * radius + dir_vector
        plane_point = ooi_points[i] + plane_vector

        mask_filter = ((point_cloud[neighbor_list] - plane_point) @ dir_norm) > 0
        neighbors_filtered = neighbor_list[mask_filter]
        refined_mask[neighbors_filtered] = True

    point_rating = refined_mask.astype(float)
    return point_rating


# ------------------------
def compute_radius(density_metric: np.ndarray, c: float):
    med = np.median(density_metric)
    mad = np.median(np.abs(density_metric - med))
    r = med + c * mad
    return r

# ------------------------
# https://huggingface.co/ZhengPeng7/BiRefNet
# Ref: Zheng et al. (2024) – BiRefNet segmentation
def extract_objects(birefnet, image_paths):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    masks = []
    for image_path in tqdm(image_paths, desc="  salient object detection"):
        image = Image.open(image_path)
        input_images = transform_image(image).unsqueeze(0).to(device)
        if device.type == 'cuda':
            input_images = input_images.half()

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu() #TODO check .cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        masks.append(np.array(mask))
    return  masks