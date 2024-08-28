# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: delete once released in MONAI

from typing import Union

import torch
import numpy as np

from monai.metrics.utils import do_metric_reduction, ignore_background
from nyumets.metrics.utils import is_binary_tensor
from monai.utils import MetricReduction

from monai.metrics.metric import CumulativeIterationMetric


class IoUPerClass(CumulativeIterationMetric):
    """
    Modified "MeanIou" for per-class (i.e. per tumor) use case.
    TODO: update documentaton below

    Compute average IoU score between two tensors. It can support both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_pred` is expected to have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
    in ``monai.transforms.post`` first to achieve binarized values.
    The `include_background` parameter can be set to ``False`` to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background.
    `y_pred` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.
    Args:
        include_background: whether to skip IoU computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.
    """

    def __init__(
        self,
        include_background: bool = True,
        is_onehot: bool = False,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.is_onehot = is_onehot
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean IoU metric. It must be one-hot format and first dim is batch.
                The values should be binarized.
        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        #is_binary_tensor(y_pred, "y_pred")
        #is_binary_tensor(y, "y")

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        
        return compute_iouperclass(
            y_pred=y_pred, y=y,
            include_background=self.include_background,
            ignore_empty=self.ignore_empty,
            is_onehot=self.is_onehot
        )

    def aggregate(self, reduction: Union[MetricReduction, str, None] = MetricReduction.MEAN):  # type: ignore
        """
        Execute reduction logic for the output of `compute_meaniou`.
        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_iouperclass(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    ignore_empty: bool = True,
    is_onehot: bool = False
) -> torch.Tensor:
    """Computes IoU score metric from full size Tensor and collects average.
    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean IoU metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip IoU computation on the first channel of
            the predicted output. Defaults to True.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.
    Returns:
        IoU scores per batch and per class, (shape [batch_size, num_classes]).
    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    iou = []
    if is_onehot:
        num_tumors = y.shape[0] + 1
        for tumor_idx in range(1, num_tumors):
            intersection = torch.sum(y[tumor_idx] * y_pred[tumor_idx])
            y_tumor_sum, y_pred_tumor_sum = torch.sum(y[tumor_idx]), torch.sum(y_pred[tumor_idx])
            union = y_tumor_sum + y_pred_tumor_sum - intersection
            iou.append((intersection)/union)

    else:
        num_tumors = y.max().int().item() + 1
        for tumor_idx in range(1, num_tumors):
            y_tumor = (y == tumor_idx) + 0.
            y_o_tumor = (y_pred == tumor_idx) + 0.
            intersection = torch.sum(y_tumor * y_o_tumor)
            y1_o, y2_o = torch.sum(y_tumor), torch.sum(y_o_tumor)
            union = y1_o + y2_o - intersection
            iou.append((intersection)/union)
    
    mean_iou = [np.mean(np.array(iou))]
    
    return torch.Tensor([mean_iou])

