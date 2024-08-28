import os
import torch
import numpy as np
import yaml
from pathlib import Path
from skimage import measure
from typing import Optional, List
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils import optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
from monai.data import decollate_batch
from monai.data.utils import get_random_patch

from monai.transforms import (
    Compose,
    LoadImaged,
    LoadImage,
    AddChanneld,
    NormalizeIntensityd,
    NormalizeIntensity,
    RandAffined,
    EnsureTyped,
    EnsureType,
    AsDiscrete,
    AsDiscreted,
    ResizeWithPadOrCropd,
    ResizeWithPadOrCrop,
    RandSpatialCropd,
    EnsureChannelFirstd,
    Activations,
    ToDeviced,
    SpatialPadd,
    SpatialCrop,
    Rotate,
    RandRotated,
    RandRotate,
    Flip,
    RandFlipd,
    RandAxisFlipd,
    RandStdShiftIntensityd,
    RandGibbsNoised,
    RandBiasFieldd,
    RandKSpaceSpikeNoised,
    RandGaussianSmoothd,
    RandRicianNoised,
    ConcatItemsd,
    Spacingd,
    ScaleIntensityd
)

from nyumets.transforms.transforms import (
    ConvertToBinary,
    ConnectComponents,
    FixPreviousTimepoints,
    RegisterToImageAndApplyToLabel,
    MatchTumorsToReference,
    MaskBackgroundd,
)


cp, has_cp = optional_import("cupy")
cp_ndarray, _ = optional_import("cupy", name="ndarray")
cucim, has_cucim = optional_import("cucim")

with open(Path(__file__).parents[2] / "config.yaml") as f:
    config = yaml.safe_load(f)

CONFIG_RESIZE_SIZE = (config['resize_x'], config['resize_y'], config['resize_z'])
CONFIG_PATCH_SIZE = (config['patch_x'], config['patch_y'], config['patch_z'])


def get_nyumets_transforms(
    image_modalities: list = config['image_modalities'],
    resize_size: tuple = CONFIG_RESIZE_SIZE,
    patch_size: tuple = None,
    temporal: bool = False,
    intensity_augmentation: bool = False,
    spatial_augmentation: bool = False,
):
    all_keys = image_modalities + ['label']

    transforms_list = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        ResizeWithPadOrCropd(keys=all_keys, spatial_size=resize_size),
        ConcatItemsd(keys=image_modalities, name='image'),
        NormalizeIntensityd(keys=['image'], nonzero=True),
        MaskBackgroundd(keys=['label'], foreground_key='image')
    ]

    if intensity_augmentation:
        intensity_transforms = [
            RandStdShiftIntensityd(keys=['image'], prob=0.1, factors=(2, 5)),
            RandGibbsNoised(keys=['image'], prob=0.1, alpha=(0.0, 1.0)),
            RandBiasFieldd(keys=['image'], prob=0.1),
            RandRicianNoised(keys=['image'], prob=0.1, mean=0.0, std=0.5),
            RandGaussianSmoothd(keys=['image'], prob=0.1),
        ]
        transforms_list.extend(intensity_transforms)

    if not temporal:
        if spatial_augmentation:
            spatial_transforms = [
                RandRotated(keys=['image', 'label'], mode=['bilinear', 'nearest'], prob=0.1, range_x=0.4, range_y=0.4),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1)
            ]
            transforms_list.extend(spatial_transforms)
        
        if patch_size is not None:
            transforms_list.append(RandSpatialCropd(keys=['image', 'label'], roi_size=patch_size, random_size=False))

    transforms_list.append(EnsureTyped(keys=['image', 'label']))
    
    return Compose(transforms_list)


def get_brats21_transforms(
    image_modalities: list = config['image_modalities'],
    resize_size: tuple = CONFIG_RESIZE_SIZE,
    patch_size: tuple = None,
):
    all_keys = image_modalities + ['label']

    transforms_list = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        ConvertToBinary(keys=['label']),
        ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=resize_size),
        ConcatItemsd(keys=image_modalities, name='image'),
        NormalizeIntensityd(keys=['image']),
    ]

    if patch_size is not None:
        transforms_list.append(RandSpatialCropd(keys=['image', 'label'], roi_size=patch_size, random_size=False))
    
    transforms_list.append(EnsureTyped(keys=['image', 'label']))

    return Compose(transforms_list)


def get_stanfordbrainmetsshare_transforms(
    image_modalities: list = config['image_modalities'],
    resize_size: tuple = CONFIG_RESIZE_SIZE,
    patch_size: tuple = None,
    intensity_augmentation: bool = False,
    spatial_augmentation: bool = False
):
    all_keys = image_modalities + ['label']
    transforms_list = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Spacingd(keys=all_keys, pixdim=(1.,1.,1.), mode=['bilinear', 'nearest']),
        ResizeWithPadOrCropd(keys=all_keys, spatial_size=resize_size),
        ConcatItemsd(keys=image_modalities, name='image'),
        NormalizeIntensityd(keys=['image'], nonzero=True),
        ScaleIntensityd(keys=['label'], minv=0.0, maxv=1.0)
    ]

    if intensity_augmentation:
        intensity_transforms = [
            RandStdShiftIntensityd(keys=['image'], prob=0.1, factors=(2, 5)),
            RandGibbsNoised(keys=['image'], prob=0.1, alpha=(0.0, 1.0)),
            RandBiasFieldd(keys=['image'], prob=0.1),
            RandRicianNoised(keys=['image'], prob=0.1, mean=0.0, std=0.5),
            RandGaussianSmoothd(keys=['image'], prob=0.1),
        ]
        transforms_list.extend(intensity_transforms)

    if spatial_augmentation:
        spatial_transforms = [
            RandRotated(keys=['image', 'label'], mode=['bilinear', 'nearest'], prob=0.1, range_x=0.4, range_y=0.4),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1)
        ]
        transforms_list.extend(spatial_transforms)
        
    if patch_size is not None:
        patch_transforms = [
            RandSpatialCropd(keys=['image', 'label'], roi_size=patch_size, random_size=False)
        ]
        transforms_list.extend(patch_transforms)
        
    transforms_list.append(EnsureTyped(keys=['image', 'label']))
    
    return Compose(transforms_list)


def post_processing_transforms(preds, targets):
    targets_processing = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    preds_processing = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])

    preds = [preds_processing(i) for i in decollate_batch(preds)]
    targets = [targets_processing(i) for i in decollate_batch(targets)]

    return (preds, targets)


def get_cc_transforms(onehot_tumors = config['onehot_tumors']):
    # connected components
    cc_labels_processing = Compose(
        [
            EnsureType(),
            ConnectComponents(to_onehot=onehot_tumors)
        ]
    )
    cc_outputs_processing = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True),
            ConnectComponents(to_onehot=onehot_tumors)
        ]
    )
    return (cc_outputs_processing, cc_labels_processing)


def get_match_outputs_transforms(
    is_onehot = config['onehot_tumors'],
    distance_threshold = config['outputs_distance_threshold'],
    ior_threshold = config['outputs_ior_threshold'],
    keep_false_positives = config['keep_fp']
):
    match_outputs_processing = MatchTumorsToReference(
        is_onehot=is_onehot,
        distance_threshold=distance_threshold,
        ior_threshold=ior_threshold,
        keep_old=keep_false_positives
    )
    return match_outputs_processing


def cc_processing_transforms(preds, targets):
    # connected components
    (cc_outputs_processing, cc_labels_processing) = get_cc_transforms()

    # matching outputs
    match_outputs_processing = get_match_outputs_transforms()
    
    targets = [cc_labels_processing(y) for y in decollate_batch(targets)]
    preds = [cc_outputs_processing(y) for y in decollate_batch(preds)]

    preds = [match_outputs_processing(y, y_ref) for y, y_ref in zip(preds, targets)]
    
    return (preds, targets)


def get_previous_timepoint_transforms(
    resize_size = CONFIG_RESIZE_SIZE
):
    # processing for previous timepoints
    prev_img_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            ResizeWithPadOrCrop(spatial_size=resize_size),
            NormalizeIntensity(),
            EnsureType()
        ]
    )
    prev_label_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            ResizeWithPadOrCrop(spatial_size=resize_size),
            EnsureType()
        ]
    )
    fix_prev_processing = FixPreviousTimepoints()

    return (prev_img_transforms, prev_label_transforms, fix_prev_processing)

def get_match_prev_transforms(
    is_onehot = config['onehot_tumors'],
    prev_distance_threshold = config['prev_distance_threshold'],
    prev_ior_threshold = config['prev_ior_threshold'],
    keep_false_positives = config['keep_fp']
):
    match_prev_processing = MatchTumorsToReference(
        is_onehot=is_onehot,
        distance_threshold=prev_distance_threshold,
        ior_threshold=prev_ior_threshold,
        keep_old=False
    )
    return match_prev_processing


def longitudinal_transforms(preds, targets, inputs, prev_inputs, prev_targets, resize_size=CONFIG_RESIZE_SIZE):
    # get transforms
    (prev_img_transforms, prev_label_transforms, fix_prev_processing) = get_previous_timepoint_transforms(resize_size=resize_size)
    cc_labels_processing, cc_outputs_processing = get_cc_transforms()
    coregister_processing = RegisterToImageAndApplyToLabel()
    match_prev_processing = get_match_prev_transforms()
    match_outputs_processing = get_match_outputs_transforms()

    # load previous timepoint's images and labels and run same transforms as val_transforms during dataloading
    post_prev_images = [prev_img_transforms(y) for y in decollate_batch(prev_inputs)]
    post_prev_labels = [prev_label_transforms(y) for y in decollate_batch(prev_targets)]

    # register to set sequence
    # TODO: need to test
    image_modalities = config['image_modalities']
    reg_sequence_name = config['registration_sequence']
    reg_seq_ix = image_modalities.index(reg_sequence_name)
    registration_sequence_inputs = [inp[reg_seq_ix,...] for inp in decollate_batch(val_inputs)]
    registration_sequence_inputs = [torch.unsqueeze(inp, 0) for inp in registration_sequence_inputs]

    # set any images or labels which aren't previous timepoints to arrays with all -1
    post_prev_images = [fix_prev_processing(y_prev, y_curr) for y_prev, y_curr in zip(post_prev_images, registration_sequence_inputs)]
    post_prev_labels = [fix_prev_processing(y_prev, y_curr) for y_prev, y_curr in zip(post_prev_labels, decollate_batch(targets))]

    # post-process outputs and labels
    targets = [cc_labels_processing(y) for y in decollate_batch(targets)]
    preds = [cc_outputs_processing(y) for y in decollate_batch(preds)]

    # coregister outputs and labels to previous image
    preds = [coregister_processing(y_img.cpu(), y_prev_img.cpu(), y_label.cpu()) for y_img, y_prev_img, y_label in zip(registration_sequence_inputs, post_prev_images, preds)]
    targets = [coregister_processing(y_img.cpu(), y_prev_img.cpu(), y_label.cpu()) for y_img, y_prev_img, y_label in zip(registration_sequence_inputs, post_prev_images, targets)]

    # match predicted tumors to target tumors
    preds = [match_outputs_processing(y, y_ref) for y, y_ref in zip(preds, targets)]

    # match previous (groundtruth) tumors to current (groundtruth) tumors
    post_prev_labels = [match_prev_processing(y, y_ref) for y, y_ref in zip(post_prev_labels, targets)]

    return (preds, targets, post_prev_labels)


def get_roi_start_end(
    image_size,
    patch_size: tuple
):
    if len(image_size) != 5:
        raise NotImplementedError("This function only works with [B,C,H,W,D] tensors")

    spatial_size = image_size[2:]
    
    rand_int = np.random.randint
    min_corner = tuple(rand_int(0, ms - ps + 1) if ms > ps else 0 for ms, ps in zip(spatial_size, patch_size))

    start_coords = []
    end_coords = []
    for mc, ps in zip(min_corner, patch_size):
        start = mc
        end = mc + ps
        start_coords.append(start)
        end_coords.append(end)
    return tuple(start_coords), tuple(end_coords)


def patch_batch(
    image: torch.Tensor,
    label: torch.Tensor,
    patch_size: tuple = CONFIG_PATCH_SIZE,
):

    roi_start, roi_end = get_roi_start_end(image.size(), patch_size) 
    crop_fn = SpatialCrop(roi_start=roi_start, roi_end=roi_end)
    
    image_crop = torch.zeros((image.shape[0], image.shape[1], patch_size[0], patch_size[1], patch_size[2]), device=image.device)
    label_crop = torch.zeros((label.shape[0], label.shape[1], patch_size[0], patch_size[1], patch_size[2]), device=image.device)
    
    for b in range(image.shape[0]):
        image_crop[b,...] = crop_fn(image[b,...])
        label_crop[b,...] = crop_fn(label[b,...])
    
    return image_crop, label_crop


def get_flip_aug_transforms(
    flip_x_prob: float = 0.1,
    flip_y_prob: float = 0.1,
):
    do_flip_x = np.random.rand() < flip_x_prob
    do_flip_y = np.random.rand() < flip_y_prob

    if do_flip_x or do_flip_y:
        flip_axes = []
        if do_flip_x:
            flip_axes.append(0)
        if do_flip_y:
            flip_axes.append(1)

        flip_fn = Flip(spatial_axis=flip_axes)
        return flip_fn

    return None


def get_rotation_aug_transform(
    rotate_prob: float = 0.1,
    max_rotate_radians: float = 0.4
):
    do_rotate = np.random.rand() < rotate_prob

    if do_rotate:
        rand_angle_x = np.random.uniform(low=-max_rotate_radians, high=max_rotate_radians)
        rand_angle_y = np.random.uniform(low=-max_rotate_radians, high=max_rotate_radians)
        rand_angle_z = np.random.uniform(low=-max_rotate_radians, high=max_rotate_radians)

        rotate_fn = Rotate(angle=(rand_angle_x, rand_angle_y, rand_angle_z))
        return rotate_fn
    
    return None


def spatial_augment_batch(
    image: torch.Tensor,
    label: torch.Tensor,
    flip_x_prob: float = 0.1,
    flip_y_prob: float = 0.1,
    rotate_prob: float = 0.1,
    max_rotate_radians: float = 0.4,
):
    # TODO: This would be faster to do on worker
    image_aug = torch.zeros((image.shape), device=image.device)
    label_aug = torch.zeros((label.shape), device=image.device)

    flip_fn = get_flip_aug_transforms(flip_x_prob, flip_y_prob)
    rotate_fn = get_rotation_aug_transform(rotate_prob, max_rotate_radians)

    if rotate_fn is not None:
        for b in range(image.shape[0]):
            image_aug[b,...] = rotate_fn(image[b,...])
            label_aug[b,...] = rotate_fn(label[b,...])
        image = image_aug
        label = label_aug

    if flip_fn is not None:
        for b in range(image.shape[0]):
            image_aug[b,...] = flip_fn(image[b,...])
            label_aug[b,...] = flip_fn(label[b,...])
    
    return image_aug, label_aug
