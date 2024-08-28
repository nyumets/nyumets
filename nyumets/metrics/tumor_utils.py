from typing import Optional, List
import numpy as np
import torch
import skimage
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils import optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

cp, has_cp = optional_import("cupy")
cp_ndarray, _ = optional_import("cupy", name="ndarray")
cucim, has_cucim = optional_import("cucim")


### TUMOR COUNT/VOLUME HELPERS FOR METRICS

def calculate_tumor_count(
    y,
    volume_threshold: float = None,
    components_connected: bool = False,
    is_onehot: bool = False,
    spacing: list = [1., 1., 1.]
) -> int:
    """
    Calculate the total number of tumors in a segmentation array.
    """
    if not components_connected:
        y_cc, num_tumors = get_connected_components(y, return_num=True)
    else:
        if is_onehot:
            num_tumors = y.shape[0]
        else:
            y_cc = y
            num_tumors = y_cc.max().int().item()

    if volume_threshold is not None:
        for tumor_id in range(num_tumors+1):
            if is_onehot:
                tumor_volume = measure_total_tumor_volume(y[tumor_id,...], spacing=spacing)
            else:
                tumor_volume = measure_single_tumor_volume(y_cc, tumor_id=tumor_id, spacing=spacing)
            
            if tumor_volume >= volume_threshold:
                num_tumors -= 1
    
    return num_tumors


def get_connected_components(
    img: NdarrayTensor,
    connectivity: Optional[int] = None,
    return_num_only: bool = False,
    return_num: bool = False,
):
    """
    Gets connected components.

    Args:
        img: Image to connected components from. Shape is (spatial_dim1 [, spatial_dim2, ...])
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used. for more details:
            https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label.
        return_num_only: Return the number of connected components instead of the connected components array.
    """
    
    if isinstance(img, torch.Tensor) and has_cp and has_cucim:
        x_cupy = monai.transforms.ToCupy()(img.short())
        x_label, num = cucim.skimage.measure.label(x_cupy, return_num=True, connectivity=connectivity)
        img_label = monai.transforms.ToTensor(device=img.device)(x_label)

    else:
        img_arr = convert_data_type(img, np.ndarray)[0]
        img_label, num = skimage.measure.label(img_arr, return_num=True, connectivity=connectivity)
        img_label = convert_to_dst_type(img_label, dst=img, dtype=img.dtype)[0]
    
    if return_num:
        return img_label, num
    elif return_num_only:
        return num

    return img_label


def measure_total_tumor_volume(
    y,
    spacing: list = [1., 1., 1.]
) -> float:
    """
    Measure the total volume of all nonzero values in a 3D segmentation array.
    """
    units = torch.count_nonzero(y)
    volume = units * spacing[0] * spacing[1] * spacing[2]
    return volume


def measure_single_tumor_volume(
    y,
    tumor_id: int = 1,
    spacing=[1., 1., 1.]
) -> float:
    """
    Measure the volume of a single tumor in a 3D segmentation array.
    """
    #y_tumor = y[torch.where(y==tumor_id)]
    y_tumor = y[y==tumor_id]
    tumor_volume = measure_total_tumor_volume(y_tumor, spacing=spacing)
    return tumor_volume


def get_per_tumor_volumes(
    y,
    spacing=[1., 1., 1.],
    is_onehot=False,
    num_tumors=None  # number of true positive tumors
):
    if num_tumors is None:
        if is_onehot:
            num_tumors = y.shape[0]
        else:
            num_tumors = y.max().int().item()

    tumor_vol_list = torch.zeros(num_tumors)

    if is_onehot:
        for i in range(1,num_tumors):
            tumor_vol_list[i] = measure_total_tumor_volume(y[i,...], spacing=spacing)
    else:
        for i in range(1,num_tumors):
            tumor_vol_list[i] = measure_single_tumor_volume(y, tumor_id=i, spacing=spacing)

    return tumor_vol_list


### MATCHING TUMORS

def match_tumors(y, y_ref, is_onehot, distance_threshold, ior_threshold, keep_old):
    """
    labels and reference labels must already be in connected components format,
    either with each tumor labeled with an int or in onehot format with each
    channel representing a unique tumor.

    Args:
        y: labels to be matched to reference
        y_ref: reference labels
        is_onehot: bool, true if the labels is onehot-encoded
        distance_threshold: maximum distance between tumor centroids
        ior_threshold: minimum ior (intersection over reference) between tumors
        keep_old: whether to keep the unmatched tumors
    
    Returns:
        matched_img: the labels with tumors matched to reference
    """
    # get centroids of all of the tumors
    y_centroids = get_centroids(y, is_onehot)

    # if no tumors found in y, return unchanged y
    if np.array_equal(y_centroids, np.zeros((1, 3))):
        return y

    y_ref_centroids = get_centroids(y_ref, is_onehot)

    # calculate the maximum possible distance in the matrix
    max_distance = np.linalg.norm(np.array(y_ref.shape[1:]) - np.array([0,0,0]))
    # get distances between all of the pairs of centroids
    all_distances = get_all_centroid_distances(y_ref_centroids, y_centroids, max_distance)
    # get all of the iors between all pairs of tumors
    all_ior = get_all_ior(y_ref, y, is_onehot)

    # get matches based distance and iou
    dist_tumor_matches = match_tumors_by_distance(all_distances, distance_threshold)
    ior_tumor_matches = match_tumors_by_ior(all_ior, ior_threshold)

    # keep tumor matched only if they satisfy both matching criteria
    tumor_matches = np.zeros(dist_tumor_matches.shape[0])
    for i, (dist, ior) in enumerate(zip(dist_tumor_matches, ior_tumor_matches)):
        if dist == ior:
            tumor_matches[i] = dist

    tumor_matches = tumor_matches.astype(int)

    # return the matrix with reassigned labels based on the reference
    return reassign_matches(y_ref, y, tumor_matches, is_onehot, keep_old)


def get_centroids(y, is_onehot):
    if is_onehot:
        num_tumors = y.shape[0] + 1
        centroids = np.zeros((num_tumors, 3))
        for tumor_idx in range(num_tumors):
            if tumor_idx == 0:  # ignore background
                continue
            region = skimage.measure.regionprops(y[tumor_idx].int().numpy())
            if len(region) > 0:
                for i in range(len(region[0].centroid)):
                    centroids[tumor_idx,i] = region[0].centroid[i]

    else:
        num_tumors = y.max().int().item() + 1
        centroids = np.zeros((num_tumors, 3))
        for tumor_idx in range(num_tumors):
            if tumor_idx == 0:  # ignore background
                continue
            y_tumor = (y == tumor_idx) + 0.
            region = skimage.measure.regionprops(y_tumor[0].int().numpy())
            if len(region) > 0:
                for i in range(len(region[0].centroid)):
                    centroids[tumor_idx,i] = region[0].centroid[i]

    return centroids


def get_all_centroid_distances(centroids1, centroids2, max_distance = 300):
    # TODO: vectorize these operations
    n1, n2 = centroids1.shape[0], centroids2.shape[0]
    all_distances = np.ones((n1, n2)) * max_distance
    for i in range(1, n1):  # start at one to ignore background
        for j in range(1, n2):
            all_distances[i, j] = np.linalg.norm(centroids1[i] - centroids2[j])
    return all_distances


def match_tumors_by_distance(all_distances, distance_threshold = 5.):
    tumor_list = np.zeros(all_distances.shape[0])

    for i in range(all_distances.shape[0]):
        min_idx = np.argmin(all_distances[i])
        if all_distances[i, min_idx] <= np.min(all_distances[:,min_idx]) and all_distances[i, min_idx] <= distance_threshold:
            tumor_list[i] = min_idx
    
    return tumor_list.astype(int)


def get_all_ior(y, y_o, is_onehot):
    if is_onehot:
        n1, n2 = y.shape[0] + 1, y_o.shape[0] + 1
        all_ior = np.zeros((n1, n2))
        
        for i in range(1, n1):
            for j in range(1, n2):
                intersection = torch.sum(y[i] * y_o[j])
                reference = torch.sum(y[i])
                all_ior[i, j] = (intersection)/reference

    else:
        n1, n2 = y.max().int().item() + 1, y_o.max().int().item() + 1
        all_ior = np.zeros((n1, n2))
        
        for i in range(1, n1):
            for j in range(1, n2):
                y_tumor = (y == i) + 0.
                y_o_tumor = (y_o == j) + 0.
                intersection = torch.sum(y_tumor * y_o_tumor)
                reference = torch.sum(y_tumor)
                all_ior[i, j] = (intersection)/reference
        
    return all_ior


def get_all_iou(y, y_o, is_onehot):
    if is_onehot:
        n1, n2 = y.shape[0] + 1, y_o.shape[0] + 1
        all_iou = np.zeros((n1, n2))

        for i in range(1, n1):
            for j in range(1, n2):
                intersection = torch.sum(y[i] * y_o[j])
                y1_o, y2_o = torch.sum(y[i]), torch.sum(y_o[j])
                union = y1_o + y2_o - intersection
                all_iou[i, j] = (intersection)/union

    else:
        n1, n2 = y.max().int().item() + 1, y_o.max().int().item() + 1
        all_iou = np.zeros((n1, n2))
        
        for i in range(1, n1):
            for j in range(1, n2):
                y_tumor = (y == i) + 0.
                y_o_tumor = (y_o == j) + 0.
                intersection = torch.sum(y_tumor * y_o_tumor)
                y1_o, y2_o = torch.sum(y_tumor), torch.sum(y_o_tumor)
                union = y1_o + y2_o - intersection
                all_iou[i, j] = (intersection)/union
        
    return all_iou
            

def match_tumors_by_ior(all_ior, ior_threshold = 0.):
    tumor_list = np.zeros(all_ior.shape[0])
    
    for i in range(all_ior.shape[0]):
        max_idx = np.argmax(all_ior[i])
        if all_ior[i, max_idx] >= np.max(all_ior[:,max_idx]) and all_ior[i, max_idx] >= ior_threshold:
            tumor_list[i] = max_idx
    
    return tumor_list.astype(int)


def reassign_matches(y_ref, y_old, tumor_matches, is_onehot, keep_old):
    """
    Args:
        y_ref: reference labels
        y_old: labels to be matched
        tumor_matched: list of tumor matches, where the index indicates the new
            tumor id, and the value at that index indicates the old tumor id
        is_onehot: whether the matrix is onehot encoded (by tumor)
        keep_old: whether to keep the unmatched tumors (i.e. false positives)
    """

    if is_onehot and keep_old:
        matched_img = torch.zeros(y_old.shape)
        old_num_tumors = y_old.shape[0]
    else:
        matched_img = torch.zeros(y_ref.shape)
        if keep_old:
            old_num_tumors = y_old.max().int().item()

    new_id = 0

    if is_onehot:
        for new_id, old_id in enumerate(tumor_matches):
            if old_id != 0.:  # ignore background and no matched indices
                matched_img[new_id,...] = y_old[old_id,...]
        
        if keep_old:
            for old_id in range(1, old_num_tumors+1):
                if old_id not in tumor_matches:
                    new_id += 1
                    matched_img[new_id,...] = y_old[old_id,...]
    
    else:
        for new_id, old_id in enumerate(tumor_matches):
            if old_id != 0.:  # ignore background and no matched indices
                matched_img[y_old == old_id] = new_id

        if keep_old:
            for old_id in range(1, old_num_tumors+1):
                if old_id not in tumor_matches:
                    new_id += 1
                    matched_img[y_old == old_id] = new_id
    
    return matched_img

