import torch
import numpy as np
from skimage import measure

from typing import Optional, List

from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils import optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
from monai.data import decollate_batch

from nyumets.metrics.tumor_utils import (
    calculate_tumor_count, 
    measure_total_tumor_volume,
    get_per_tumor_volumes
)

# MONAI future version
# TODO: remove once this has been integrated into MONAI release
def is_binary_tensor(input: torch.Tensor, name: str):
    """Determines whether the input tensor is torch binary tensor or not.

    Args:
        input (torch.Tensor): tensor to validate.
        name (str): name of the tensor being checked.

    Raises:
        ValueError: if `input` is not a PyTorch Tensor.

    Returns:
        Union[str, None]: warning message, if the tensor is not binary. Othwerwise, None.
    """
    if not isinstance(input, torch.Tensor):
        raise ValueError(f"{name} must be of type PyTorch Tensor.")
    if not torch.all(input.byte() == input) or input.max() > 1 or input.min() < 0:
        warnings.warn(f"{name} should be a binarized tensor.")


def get_eval_metrics(
    outputs,
    labels,
    outputs_processing,
    labels_processing,
    metrics: List
):
    metrics_agg = []
    post_outputs = [outputs_processing(y) for y in decollate_batch(outputs)]
    post_labels = [labels_processing(y) for y in decollate_batch(labels)]
    for metric in metrics:
        metric(y_pred=post_outputs, y=post_labels)
        metrics_agg.append(metric.aggregate().item())
        metric.reset()
    return metrics_agg


def calculate_tumor_stats(labels, patient_id=None, study_id=None, prev_matched_labels=None, prev_study_id=None):
    tumor_stats = {
        'patient_id': patient_id,
        'study_id': study_id,
        'prev_study_id': prev_study_id,
        'tumor_count': None,
        'tumor_volume': None,
        'per_tumor_volume': None,
        'prev_tumor_count': None,
        'prev_tumor_volume': None,
        'prev_per_tumor_volume': None,
        'change_tumor_count': None,
        'change_tumor_volume': None,
        'change_per_tumor_volume': None
    }
    
    # if labels include a channel dimension
    if len(labels.size()) > 3:
        labels = labels[0,...]
    
    tumor_stats['tumor_count'] = calculate_tumor_count(labels)
    tumor_stats['tumor_volume'] = measure_total_tumor_volume(labels).item()
    tumor_stats['per_tumor_volumes'] = np.array(get_per_tumor_volumes(labels))
    
    if prev_matched_labels is not None:
        
        if len(prev_matched_labels.size()) > 3:
            prev_matched_labels = prev_matched_labels[0,...]
        
        tumor_stats['prev_tumor_count'] = calculate_tumor_count(prev_matched_labels)
        tumor_stats['prev_tumor_volume'] = measure_total_tumor_volume(prev_matched_labels).item()
        tumor_stats['prev_per_tumor_volume'] = np.array(get_per_tumor_volumes(prev_matched_labels))
        
        tumor_stats['change_tumor_count'] = tumor_stats['tumor_count'] - tumor_stats['prev_tumor_count']
        tumor_stats['change_tumor_volume'] = tumor_stats['tumor_volume'] - tumor_stats['prev_tumor_volume']
        tumor_stats['change_per_tumor_volume'] = np.subtract(tumor_stats['per_tumor_volumes'], tumor_stats['prev_per_tumor_volume'])
        tumor_stats['percent_change_per_tumor_volume'] = np.divide(tumor_stats['change_per_tumor_volume'], tumor_stats['prev_per_tumor_volume'])
    
    return tumor_stats