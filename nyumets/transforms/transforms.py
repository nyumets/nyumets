import numpy as np
from typing import Any, Callable, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, Dict

import torch
import SimpleITK as sitk
from skimage import measure

from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform, LoadImage, EnsureChannelFirst

from monai.utils import (
    TransformBackends,
    convert_data_type,
    convert_to_tensor,
    deprecated_arg,
    ensure_tuple,
    look_up_option,
)
from monai.utils.type_conversion import convert_to_dst_type

from nyumets.data.meta_obj import get_track_meta
from nyumets.metrics.tumor_utils import match_tumors


class ConvertToBinary(MapTransform):
    """
    Convert labels to multi channels based on BraTS classes:
    label 1 is the necrotic tumor core
    label 2 is the peritumoral edematous / invaded tissue
    label 4 is the GD-enhancing tumor
    For NYUMets, the possible classes are total tumor = label 1 and label 4
    and not tumor = label 0 and label 2
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # merge label 1 and label 4 to construct total tumor
            d[key] = np.logical_or(d[key] == 1, d[key] == 4).astype(np.float32)
        return d


# From MONAI
# TODO: delete once integrated into release
def unique(x: NdarrayTensor) -> NdarrayTensor:
    """`torch.unique` with equivalent implementation for numpy.

    Args:
        x: array/tensor
    """
    return torch.unique(x) if isinstance(x, torch.Tensor) else np.unique(x)  # type: ignore


def get_unique_labels(
    img: NdarrayOrTensor, is_onehot: bool, discard: Optional[Union[int, Iterable[int]]] = None
) -> Set[int]:
    """Get list of non-background labels in an image.

    Args:
        img: Image to be processed. Shape should be [C, W, H, [D]] with C=1 if not onehot else `num_classes`.
        is_onehot: Boolean as to whether input image is one-hotted. If one-hotted, only return channels with
        discard: Can be used to remove labels (e.g., background). Can be any value, sequence of values, or
            `None` (nothing is discarded).

    Returns:
        Set of labels
    """
    applied_labels: Set[int]
    n_channels = img.shape[0]
    if is_onehot:
        applied_labels = {i for i, s in enumerate(img) if s.sum() > 0}
    else:
        if n_channels != 1:
            raise ValueError("If input not one-hotted, should only be 1 channel.")
        applied_labels = set(unique(img).tolist())
    if discard is not None:
        for i in ensure_tuple(discard):
            applied_labels.discard(i)
    return applied_labels


class ConnectComponents(Transform):
    """
    """
    def __init__(
        self,
        to_onehot: bool = True,
        connectivity: Optional[int] = None,
    ):
        super().__init__()
        self.to_onehot = to_onehot
        self.connectivity = connectivity
        
    def __call__(self, labels):

        if torch.all(torch.eq(labels, -1)):
            # no labels available
            return labels

        # ignore background
        labels = labels[1:,...] if labels.shape[0] > 1 else labels
        device = labels.device
        
        labels_arr = convert_data_type(labels, np.ndarray)[0]
        labels_cc, num_labels = measure.label(labels_arr[0], return_num=True, connectivity=self.connectivity)
        labels_cc = labels_cc[None,...]

        if self.to_onehot:
            labels_onehot = np.zeros((num_labels+1, labels_cc.shape[1], labels_cc.shape[2], labels_cc.shape[3]))
            for label in range(num_labels+1):
                labels_onehot[label,...] = (labels_cc == label) + 0.
            return torch.Tensor(labels_onehot).to(device)
                
        return torch.Tensor(labels_cc).to(device) 

class MatchTumorsToReference(Transform):
    """
    Labels and reference labels must be connected components, either in ints or
    onehot-encoded format. See ConnectComponents() transform above.
    """
    def __init__(
        self,
        is_onehot: bool = True,
        distance_threshold: float = None,
        ior_threshold: float = None,
        keep_old: bool = True
    ):
        super().__init__()
        self.is_onehot = is_onehot
        self.distance_threshold = distance_threshold
        self.ior_threshold = ior_threshold
        self.keep_old = keep_old
        
    def __call__(self, labels, reference_labels):

        if torch.all(torch.eq(labels, -1)):
            # no tumors to match, keep as-is
            return labels

        matched_img = match_tumors(
            labels,
            reference_labels,
            self.is_onehot,
            self.distance_threshold,
            self.ior_threshold,
            self.keep_old
        )
        return matched_img


class MaskBackground(Transform):
    def __init__(
        self,
        background_pixel_value: float = 0.
    ):
        """
        Mask any part of image that is outside the foreground (e.g. segmentations outside of a brain mask).
        Ignores batch and channel dimensions.

        Args:
            background_pixel_value: value of pixels in background. Default is 0.
        """
        super().__init__()
        self.background_pixel_value = background_pixel_value
    
    def __call__(self, image_to_mask, foreground_image):
        """
        Ignores batch and channel dimensions.
        
        Args:
            image_to_mask: image to apply mask to
            foreground_image: image to create mask from. HW[D] dims should be same size as image_to_mask.
        """
        
        if image_to_mask.shape != foreground_image.shape:
            raise ValueError(f'`image_to_mask` ({image_to_mask.shape}) must be the same size as `foreground_image` ({foreground_image.shape[2:]}).')
        
        image_to_mask[foreground_image == self.background_pixel_value] = self.background_pixel_value

        return image_to_mask

    
class MaskBackgroundd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        foreground_key: str,
        foreground_channel_idx: int = 0,
        background_pixel_value: float = 0.,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            foreground_key: key of image containing foreground/background.
            foreground_channel_idx: index of channel that will be used to define foreground. Default is 0.
            background_pixel_value: value of pixels in background. Default is 0.
        """
        super().__init__(keys, allow_missing_keys)
        self.foreground_key = foreground_key
        self.foreground_channel_idx = foreground_channel_idx
        self.background_pixel_value = background_pixel_value
        self.mask_background_func = MaskBackground(background_pixel_value=self.background_pixel_value)
        
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        
        d: Dict = dict(data)
               
        for key in self.key_iterator(d):
            for ch in range(d[key].shape[0]):
                d[key][ch,...] = self.mask_background_func(image_to_mask=d[key][ch,...], foreground_image=d[self.foreground_key][self.foreground_channel_idx,...])
        
        return d


##### COREGISTRATION #####

class RegisterToImaged(MapTransform):
    """
  
    """
    def __init__(
        self,
        keys: KeysCollection,
        fixed_image_key: str = 'prev_image',
        #key_apply: str = 'label',
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            key: key of (moving) image to be transformed.
            ref_index: name of key of reference (fixed) image.
            keys_apply: key(s) to apply the output transform between the 
        """
        
        super().__init__(keys, allow_missing_keys)
        self.ref_key = fixed_image_key
        #self.key_apply = key_apply
        self.register = CoregisterImages()
    
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """
        
        """
        d: Dict = dict(data)

        for key in self.key_iterator(d):
            if d[self.ref_key] is None:
                continue
            
            out_transform = calculate_coregister_transform(
                img_moving=d[key],
                img_fixed=img_fixed,
            )
            
            d[key] = coregister_images(d[key], img_fixed, out_transform)
            #d[self.key_apply] = coregister_images(d[self.key_apply], img_fixed, out_transform)
        
        return d


class RegisterToImageAndApplyToLabel(Transform):
    """
  
    """
    def __init__(
        self,
        is_label: bool = True
    ):
        """
        """
        super().__init__()
        self.is_label = is_label
    
    def __call__(self, img_moving, img_fixed, label_apply):
        """
        
        """
        # don't register if previous timepoints are -1 (i.e. no previous timepoint exists)
        if torch.all(torch.eq(img_fixed, -1)):
            return label_apply
         
        out_transform = calculate_coregister_transform(
            img_moving=img_moving,
            img_fixed=img_fixed,
        )
        
        label_apply = coregister_images(
            img_moving=label_apply,
            img_fixed=img_fixed, 
            transform=out_transform,
            is_label=self.is_label)
        
        return label_apply
    
    
class SaveCoregistrationTransformd(MapTransform):
    """
  
    """
    def __init__(
        self,
        keys: KeysCollection,
        fixed_image_key: str = 'prev_image',
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            key: key of (moving) image to be transformed.
            ref_index: name of key of reference (fixed) image.
            keys_apply: key(s) to apply the output transform between the 
        """
        
        super().__init__(keys, allow_missing_keys)
        self.ref_key = fixed_image_key
    
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """
        
        """
        d: Dict = dict(data)

        for key in self.key_iterator(d):
    
            if np.array_equal(d[self.ref_key], d[key]):
                d['image_registration_transform'] = "no transform"
                continue
            
            out_transform = calculate_coregister_transform(
                img_fixed=d[self.ref_key],
                img_moving=d[key]
            )
        
            d['image_registration_transform'] = out_transform
        
        return d


class ApplyCoregistrationTransform(Transform):
    """
    """
    def __init__(
        self,
    ):
        super().__init__()
        
    def __call__(self, img_moving, img_fixed, transform):
        
        if transform != "no transform":
            return coregister_images(img_moving, img_fixed, transform)
        else:
            return img_moving

    
class ApplyCoregisterationTransformd(MapTransform):
    """
    """
    def __init__(
        self,
        keys: KeysCollection,
        fixed_image_key: str = 'prev_image',
        transform_key: str = 'image_registration_transform',
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.transform_key = transform_key
        self.fixed_image_key = fixed_image_key
        self.apply_coreg_transform = ApplyCoregistrationTransform()
        
    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        
        d: Dict = dict(data)
        
        transform = d[self.transform_key]
        img_fixed = d[self.fixed_image_key]

        for key in self.key_iterator(d):
            
            d[key] = self.apply_coreg_transform(d[key], img_fixed, transform)
        
        return d


class FixPreviousTimepoints(Transform):
    """
    Fix any timepoints which do not have previous images by setting the previous
    image to all zeros.
    """
    def __init__(
        self,
    ):
        super().__init__()
    
    def __call__(self, prev_image, current_image):
        
        if torch.all(torch.eq(prev_image, current_image)):
            return torch.ones(prev_image.shape) * -1
                
        return prev_image


class FixPreviousTimepointsd(MapTransform):
    """
    Fix any timepoints which do not have previous images by setting the previous
    image to all zeros.
    """
    def __init__(
        self,
        keys: KeysCollection,
        ref_image_key: str = 'image',
        ref_label_key: str = 'label',
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.ref_image_key = ref_image_key
        self.ref_label_key = ref_label_key
    
    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        
        d: Dict = dict(data)
            
        for key in self.key_iterator(d):
            if np.array_equal(d[key], d[self.ref_image_key]) or np.array_equal(d[key], d[self.ref_label_key]):
                d[key] = torch.ones(d[key].shape) * -1
                
        return d


def calculate_coregister_transform(img_moving, img_fixed):
    """
    Args:
    """
    
    dims = len(img_moving.shape)
    if dims > 3:
        n_chans = img_moving.shape[0]
        if n_chans == 1:
            img_moving = torch.squeeze(img_moving, dim=0)
        else:
            raise ValueError(f"Too many channels: {n_chans}. Must be 1.")
            
    dims = len(img_fixed.shape)
    if dims > 3:
        n_chans = img_fixed.shape[0]
        if n_chans == 1:
            img_fixed = torch.squeeze(img_fixed, dim=0)
        else:
            raise ValueError(f"Too many channels: {n_chans}. Must be 1.")
            
    img_moving = sitk.GetImageFromArray(img_moving)
    img_fixed = sitk.GetImageFromArray(img_fixed)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(img_fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    out_transform = R.Execute(img_fixed, img_moving)

    return out_transform
    

def coregister_images(img_moving, img_fixed, transform=None, is_label=True):
    """
    """
    
    dims = len(img_fixed.shape)
    if dims == 4:
        n_chans = img_fixed.shape[0]
        if n_chans == 1:
            img_fixed = torch.squeeze(img_fixed, dim=0)
        else:
            raise ValueError(f"Fixed image has too many channels: {n_chans}. Must be 1.")
    else:
        raise ValueError(f"Fixed image has too many dimensions: {dims}. Must be 4 or less.")
            
    dims = len(img_moving.shape)
    if dims == 4:
        n_chans = img_moving.shape[0]
        if n_chans == 1:
            img_moving = torch.squeeze(img_moving, dim=0)
        elif n_chans == 2 and is_label:
            img_moving = img_moving[1,...]  # select the non-background dimension
            img_moving = torch.squeeze(img_moving, dim=0)
        else:
            raise ValueError(f"Moving image has too many channels: {n_chans}. Must be either 1 or 2.")
    else:
        raise ValueError(f"Moving image has too many dimensions: {dims}. Must be 4 or less.")
    
    if transform is None:
        transform = calculate_coregister_transform(img_moving, img_fixed)

    img_moving = sitk.GetImageFromArray(img_moving)
    img_fixed = sitk.GetImageFromArray(img_fixed)
    
    if is_label:
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(img_fixed)
        resample.SetTransform(transform)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        img_reg = resample.Execute(img_moving)
    
    else:
        img_reg = sitk.Resample(
            img_moving,
            img_fixed,
            transform,
            sitk.sitkLinear,
            0.0,
            img_moving.GetPixelID()
        )
    
    img_reg = sitk.GetArrayFromImage(img_reg)
    
    # Add channel dimension back
    if dims == 4:
        img_reg = img_reg[None,...]
    
    return torch.Tensor(img_reg)
