import torch
import numpy as np

from monai.data import MetaTensor

from typing import Union, Optional
from monai.utils import MetricReduction
from monai.metrics import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.metrics.regression import compute_mean_error_metrics
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

from nyumets.metrics.tumor_utils import (
    calculate_tumor_count,
    measure_total_tumor_volume, 
    measure_single_tumor_volume,
    get_per_tumor_volumes
)
from nyumets.metrics.metric import CumulativeLongitudinalMetric


class TumorCount(CumulativeIterationMetric):
    """
    """
    def __init__(
        self,
        components_connected: bool = False,
        is_onehot: bool = False,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        volume_threshold: float = None,
        abs_func = torch.abs,
        get_not_nans: bool = False,
    ):
        super().__init__()
        self.components_connected = components_connected
        self.is_onehot = is_onehot
        self.reduction = reduction
        self.volume_threshold = volume_threshold
        self.abs_func = abs_func
        self.get_not_nans = get_not_nans
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        """
        
        #is_binary_tensor(y_pred, "y_pred")
        #is_binary_tensor(y, "y")
        
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        
        y_count = []
        y_pred_count = []
        batch_size = y.shape[0]

        kwargs = {
            'volume_threshold': self.volume_threshold,
            'components_connected': self.components_connected,
            'is_onehot': self.is_onehot
        }

        for batch in range(batch_size):
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

            y_count.append(calculate_tumor_count(y[batch, 0, ...], **kwargs))
            y_pred_count.append(calculate_tumor_count(y_pred[batch, 0, ...], **kwargs))
        
        y_count = torch.Tensor([y_count])
        y_pred_count = torch.Tensor([y_pred_count])
        
        return compute_mean_error_metrics(y_pred_count, y_count, func=self.abs_func)

    
    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor) and not isinstance(data, MetaTensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f



class TumorVolume(CumulativeIterationMetric):
    """
    """
    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        abs_func = torch.abs,
        get_not_nans: bool = False,
    ):
        super().__init__()
        self.reduction = reduction
        self.abs_func = abs_func
        self.get_not_nans = get_not_nans
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        """
        
        #is_binary_tensor(y_pred, "y_pred")
        #is_binary_tensor(y, "y")
        
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        
        y_vol = []
        y_pred_vol = []
        batch_size = y.shape[0]
        
        for batch in range(batch_size):
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
            y_vol.append(measure_total_tumor_volume(y[batch, 0, ...]))
            y_pred_vol.append(measure_total_tumor_volume(y_pred[batch, 0, ...]))
        
        y_vol = torch.Tensor([y_vol])
        y_pred_vol = torch.Tensor([y_pred_vol])
        
        return compute_mean_error_metrics(y_pred_vol, y_vol, func=self.abs_func)

    
    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor) and not isinstance(data, MetaTensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


class ChangeTumorCount(CumulativeLongitudinalMetric):
    """
    """
    def __init__(
        self,
        components_connected: bool = False,
        is_onehot: bool = False,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        volume_threshold: float = None,
        abs_func = torch.abs,
        get_not_nans: bool = False,
    ):
        super().__init__()
        self.components_connected = components_connected
        self.is_onehot = is_onehot
        self.reduction = reduction
        self.volume_threshold = volume_threshold
        self.abs_func = abs_func
        self.get_not_nans = get_not_nans
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, y_prev: torch.Tensor):  # type: ignore
        """
        """
        
        #is_binary_tensor(y_pred, "y_pred")
        #is_binary_tensor(y, "y")
        
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        
        y_count = []
        y_pred_count = []
        
        batch_size = y.shape[0]

        kwargs = {
            'volume_threshold': self.volume_threshold,
            'components_connected': self.components_connected,
            'is_onehot': self.is_onehot
        }

        for batch in range(batch_size):
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
            y_prev = y_prev[:, 1:] if y_prev.shape[1] > 1 else y_prev

            if torch.all(torch.eq(y_prev, -1)):
                y_count.append(np.nan)
                y_pred_count.append(np.nan)
                continue
            
            y_prev_count = calculate_tumor_count(y_prev[batch, 0, ...], **kwargs)
            
            # subtract the previous tumor count
            y_count.append(calculate_tumor_count(y[batch, 0, ...], **kwargs) - y_prev_count)
            y_pred_count.append(calculate_tumor_count(y_pred[batch, 0, ...], **kwargs) - y_prev_count)
            
        y_count = torch.Tensor([y_count])
        y_pred_count = torch.Tensor([y_pred_count])
        
        return compute_mean_error_metrics(y_pred_count, y_count, func=self.abs_func)

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor) and not isinstance(data, MetaTenso):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


class ChangeTumorVolume(CumulativeLongitudinalMetric):
    """
    """
    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        volume_threshold: float = None,
        abs_func = torch.abs,
        get_not_nans: bool = False,
    ):
        super().__init__()
        self.reduction = reduction
        self.volume_threshold = volume_threshold
        self.abs_func = abs_func
        self.get_not_nans = get_not_nans
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, y_prev: torch.Tensor):  # type: ignore
        """
        """
        
        #is_binary_tensor(y_pred, "y_pred")
        #is_binary_tensor(y, "y")
        
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        
        y_vol = []
        y_pred_vol = []
        batch_size = y.shape[0]
        
        for batch in range(batch_size):
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
            y_prev = y_prev[:, 1:] if y_prev.shape[1] > 1 else y_prev

            if torch.all(torch.eq(y_prev, -1)):
                y_vol.append(np.nan)
                y_pred_vol.append(np.nan)
                continue
            
            y_prev_vol_batch = measure_total_tumor_volume(y_prev[batch, 0, ...])
            y_vol_batch = measure_total_tumor_volume(y[batch, 0, ...])
            y_pred_vol_batch = measure_total_tumor_volume(y_pred[batch, 0, ...])
            
            # subtract the previous tumor volume
            y_vol.append(y_vol_batch - y_prev_vol_batch)
            y_pred_vol.append(y_pred_vol_batch - y_prev_vol_batch)
        
        y_vol = torch.Tensor([y_vol])
        y_pred_vol = torch.Tensor([y_pred_vol])
        
        return compute_mean_error_metrics(y_pred_vol, y_vol, func=self.abs_func)

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor) and not isinstance(data, MetaTenso):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


class PerTumorVolume(CumulativeIterationMetric):
    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        abs_func = torch.abs,
        get_not_nans: bool = False,
        is_onehot: bool = False
    ):
        super().__init__()
        self.reduction = reduction
        self.abs_func = abs_func
        self.get_not_nans = get_not_nans
        self.is_onehot = is_onehot
        
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
            
        y_volumes = []
        y_pred_volumes = []
        batch_size = y.shape[0]
        
        for batch in range(batch_size):
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

            # get number of true positive tumors
            if self.is_onehot:
                num_tumors = y.shape[0]
            else:
                num_tumors = y.max().int().item()

            y_vols = get_per_tumor_volumes(y[batch, 0, ...], is_onehot=self.is_onehot, num_tumors=num_tumors)
            y_pred_vols = get_per_tumor_volumes(y_pred[batch, 0, ...], is_onehot=self.is_onehot, num_tumors=num_tumors)

            y_volumes.append(np.array(y_vols))
            y_pred_volumes.append(np.array(y_pred_vols))
            
        return compute_mean_error_metrics(torch.Tensor([y_pred_volumes]), torch.Tensor([y_volumes]), func=self.abs_func)

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor) and not isinstance(data, MetaTenso):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f

class ChangePerTumorVolume(CumulativeLongitudinalMetric):
    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        abs_func = torch.abs,
        get_not_nans: bool = False,
        is_onehot: bool = False
    ):
        super().__init__()
        self.reduction = reduction
        self.abs_func = abs_func
        self.get_not_nans = get_not_nans
        self.is_onehot = is_onehot
        
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, y_prev: torch.Tensor):  # type: ignore
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
            
        y_volumes = []
        y_pred_volumes = []
        batch_size = y.shape[0]
        
        for batch in range(batch_size):
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
            y_prev = y_prev[:, 1:] if y_prev.shape[1] > 1 else y_prev

            if torch.all(torch.eq(y_prev, -1)):
                y_volumes.append(np.nan)
                y_pred_volumes.append(np.nan)
                continue

            # get number of true positive tumors
            if self.is_onehot:
                num_tumors = y.shape[0]
            else:
                num_tumors = y.max().int().item()
            
            # calculate the volumes of all tumors
            y_vols_batch = get_per_tumor_volumes(y[batch, 0, ...], is_onehot=self.is_onehot, num_tumors=num_tumors)
            y_pred_vols_batch = get_per_tumor_volumes(y_pred[batch, 0, ...], is_onehot=self.is_onehot, num_tumors=num_tumors)
            y_prev_vols_batch = get_per_tumor_volumes(y_prev[batch, 0, ...], is_onehot=self.is_onehot, num_tumors=num_tumors)
            
            # subtract the previous tumor volumes
            y_volumes.append(np.subtract(np.array(y_vols_batch), np.array(y_prev_vols_batch)))
            y_pred_volumes.append(np.subtract(np.array(y_pred_vols_batch), np.array(y_prev_vols_batch)))
            
        return compute_mean_error_metrics(torch.Tensor([y_pred_volumes]), torch.Tensor([y_volumes]), func=self.abs_func)

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor) and not isinstance(data, MetaTenso):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f

class FBeta(CumulativeIterationMetric):
    """
    Calculates the Fbeta score from two instance (connected components) segmentations.
    """
    def __init__(
        self,
        beta: float = 1.,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        abs_func = torch.abs,
        get_not_nans: bool = False,
        is_onehot: bool = False
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.abs_func = abs_func
        self.get_not_nans = get_not_nans
        self.is_onehot = is_onehot
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        
        if self.is_onehot:
            
            y_pred_tumors = np.arange(1,y_pred.shape[0]+1)
            y_tumors = np.arange(1,y.shape[0]+1)
            
            # base case: y_pred is empty
            tp = 0
            fn = y.shape[0]
            fp = 0
            
            for tumor_id in y_tumors:
                if torch.count_nonzero(y_pred[tumor_id]):
                    tp += 1
                    fn -= 1    
            
            for tumor_id in np.arange(y_tumors[-1], y_pred_tumors[-1]):
                if torch.count_nonzero(y_pred[tumor_id]):
                    fp += 1
    
        else:
            
            y_pred_tumors = torch.unique(y_pred)
            y_tumors = torch.unique(y)
            
            # base case: y_pred is empty
            tp = 0
            fn = y_tumors.shape[0] - 1
            fp = 0
            
            for tumor_id in y_tumors:
                if tumor_id in y_pred_tumors and tumor_id != 0.:
                    tp += 1
                    fn -= 1
                    
            for tumor_id in y_pred_tumors:
                if tumor_id not in y_tumors:
                    fp += 1

        try:
            fbeta = [((1 + self.beta)**2 * tp) / (((1 + self.beta)**2 * tp) + (self.beta**2 * fn) + fp)]
        except ZeroDivisionError:
            fbeta = [0.]
            
        return torch.Tensor([fbeta])

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor) and not isinstance(data, MetaTenso):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f
