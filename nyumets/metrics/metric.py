import torch
from typing import Union, Optional
from abc import abstractmethod
from monai.metrics.metric import Metric, Cumulative
from monai.config import TensorOrList


class LongitudinalMetric(Metric):
    def __call__(self, y_pred: TensorOrList, y: TensorOrList, y_prev: TensorOrList):  # type: ignore
        """
        Execute basic computation for model prediction `y_pred` and ground truth `y` (optional).
        It supports inputs of a list of "channel-first" Tensor and a "batch-first" Tensor.

        Args:
            y_pred: the raw model prediction data at one iteration, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.
            y: the ground truth to compute, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.

        Returns:
            The computed metric values at the iteration level.
            The output shape could be a `batch-first` tensor or a list of `batch-first` tensors.
            When it's a list of tensors, each item in the list can represent a specific type of metric.

        """
        ret: TensorOrList
        # handling a list of channel-first data
        if isinstance(y_pred, (list, tuple)) or isinstance(y, (list, tuple)) or isinstance(y_prev, (list, tuple)):
            return self._compute_list(y_pred, y, y_prev)
        # handling a single batch-first data
        if isinstance(y_pred, torch.Tensor):
            y_ = y.detach() if isinstance(y, torch.Tensor) else None
            return self._compute_tensor(y_pred.detach(), y_, y_prev)
        raise ValueError("y_pred or y must be a list/tuple of `channel-first` Tensors or a `batch-first` Tensor.")
    
    def _compute_list(self, y_pred: TensorOrList, y: TensorOrList, y_prev: TensorOrList):
        """
        Execute the metric computation for `y_pred` and `y` in a list of "channel-first" tensors.

        The return value is a "batch-first" tensor, or a list of "batch-first" tensors.
        When it's a list of tensors, each item in the list can represent a specific type of metric values.

        For example, `self._compute_tensor` may be implemented as returning a list of `batch_size` items,
        where each item is a tuple of three values `tp`, `fp`, `fn` for true positives, false positives,
        and false negatives respectively. This function will return a list of three items,
        (`tp_batched`, `fp_batched`, `fn_batched`), where each item is a `batch_size`-length tensor.

        Note: subclass may enhance the operation to have multi-thread support.
        """
        ret = [self._compute_tensor(p.detach().unsqueeze(0), y_.detach().unsqueeze(0), y_prev_.detach().unsqueeze(0)) for p, y_, y_prev_ in zip(y_pred, y, y_prev)]
            
        # concat the list of results (e.g. a batch of evaluation scores)
        if isinstance(ret[0], torch.Tensor):
            return torch.cat(ret, dim=0)
        # the result is a list of sequence of tensors (e.g. a batch of multi-class results)
        if isinstance(ret[0], (list, tuple)) and all(isinstance(i, torch.Tensor) for i in ret[0]):
            return [torch.cat(batch_i, dim=0) for batch_i in zip(*ret)]
        return ret
    
    @abstractmethod
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, y_prev: torch.Tensor):
        """
        Computation logic for `y_pred` and `y` of an iteration, the data should be "batch-first" Tensors.
        A subclass should implement its own computation logic.
        The return value is usually a "batch_first" tensor, or a list of "batch_first" tensors.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class CumulativeLongitudinalMetric(Cumulative, LongitudinalMetric):
    """
    Base class of cumulative metric which collects metrics on each mini-batch data at the iteration level.

    Typically, it computes some intermediate results for each iteration, adds them to the buffers,
    then the buffer contents could be gathered and aggregated for the final result when epoch completed.

    For example, `MeanDice` inherits this class and the usage is as follows:

    .. code-block:: python

        dice_metric = DiceMetric(include_background=True, reduction="mean")

        for val_data in val_loader:
            val_outputs = model(val_data["img"])
            val_outputs = [postprocessing_transform(i) for i in decollate_batch(val_outputs)]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_data["seg"])  # callable to add metric to the buffer

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()

        # reset the status for next computation round
        dice_metric.reset()

    And to load `predictions` and `labels` from files, then compute metrics with multi-processing, please refer to:
    https://github.com/Project-MONAI/tutorials/blob/master/modules/compute_metric.py.

    """

    def __call__(self, y_pred: TensorOrList, y: TensorOrList, y_prev: TensorOrList):  # type: ignore
        """
        Execute basic computation for model prediction and ground truth.
        It can support  both `list of channel-first Tensor` and `batch-first Tensor`.
        Users call this API to execute computation on every batch of data, then accumulate the results,
        or accumulate the original `y_pred` and `y`, then execute on the accumulated data.

        Args:
            y_pred: the model prediction data to compute, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.
            y: the ground truth to compute, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.

        Returns:
            The computed metric values at the iteration level.
        """
        ret = super().__call__(y_pred=y_pred, y=y, y_prev=y_prev)
        if isinstance(ret, (tuple, list)):
            self.extend(*ret)
        else:
            self.extend(ret)

        return ret