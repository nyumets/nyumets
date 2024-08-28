import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import yaml

import torch

from monai.networks.layers import Norm

from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet, VNet, BasicUNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import SimpleInferer, SlidingWindowInferer

from nyumets.data.dataset import TemporalIterableDataset, TemporalShuffleBuffer
from nyumets.data.utils import (
    get_nyumets_data,
    get_brats21_data,
    get_stanfordbrainmetsshare_data
)
from nyumets.transforms.utils import (
    get_nyumets_transforms,
    get_brats21_transforms,
    get_stanfordbrainmetsshare_transforms,
    post_processing_transforms,
    cc_processing_transforms,
    longitudinal_transforms,
    patch_batch,
    spatial_augment_batch
)
from nyumets.networks.nets.sttunet import STTUNet
from nyumets.losses.utils import get_loss_function
from nyumets.metrics.tumor import (
    TumorCount,
    TumorVolume,
    PerTumorVolume,
    ChangeTumorCount,
    ChangeTumorVolume,
    ChangePerTumorVolume,
    FBeta
)
from nyumets.metrics.iou import IoUPerClass


with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml')) as f:
    config = yaml.safe_load(f)

set_determinism(seed=config['random_seed'])

parser = argparse.ArgumentParser()
parser.add_argument("--use_wandb", type=bool, default=config['use_wandb'], help="Use Weights and Biases to perform hyperparameter sweep")
parser.add_argument("--dataset", type=str, default=config['dataset'], help="Name of dataset for training. Options: 'nyumets', 'brats21', 'stanfordmets'")
parser.add_argument("--epochs", type=int, default=config['epochs'])
parser.add_argument("-lr", type=float, default=config['lr'])
parser.add_argument("--batch_size", type=int, default=config['batch_size'])  # NOTE: arg will be ignored if training STT-UNet
parser.add_argument("--model", type=str, default=config['model'], help="Model architecture to train. Options: 'vnet', 'unet', 'stt_unet")
parser.add_argument("--ckpt_dir", type=str, default=config['ckpt_dir'], help="Directory to save checkpoints. Defaults to None (will not save checkpoint)")
parser.add_argument("--calculate_extended_metrics", type=bool, default=config['always_calculate_extended_metrics'],)
parser.add_argument("--val_interval", type=int, default=config['val_interval'])
parser.add_argument("--buffer_size", type=int, default=config['buffer_size'])  # only for STT-UNet
parser.add_argument("--sequence_limit", type=int, default=config['sequence_limit'])  # only for STT-UNet
parser.add_argument("--use_patches", type=bool, default=config['use_patches'])
parser.add_argument("--spatial_augmentation", type=bool, default=config['spatial_augmentation'])
parser.add_argument("--intensity_augmentation", type=bool, default=config['intensity_augmentation'])
parser.add_argument("--debug_subset", type=int, default=None)
parser.add_argument("--use_sliding_window_inferer", type=bool, default=config['use_sliding_window_inferer'])
parser.add_argument("--loss_function", type=str, default=config['loss_function'])

def main(args):

    if args.use_wandb:
        print('WandB support not (yet) implemented.')

    device = torch.device("cuda:0")

    if args.model.lower() == 'vnet':
        model = VNet().to(device)
        batch_size = args.batch_size
        use_temporal = False
    elif args.model.lower() == 'unet':
        UNet_metadata = dict(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
        model = UNet(**UNet_metadata).to(device)
        batch_size = args.batch_size
        use_temporal = False
    elif args.model.lower() == 'unet_sttbackbone':
        model = BasicUNet().to(device)
        batch_size = args.batch_size
        use_temporal = False
    elif args.model.lower() == 'stt_unet':
        model = STTUNet().to(device)
        batch_size = 1
        use_temporal = True
    else:
        raise NotImplementedError(f"Sorry, the model '{args.model}' is not supported.")

    if args.use_patches:
        patch_size = (config['patch_x'], config['patch_y'], config['patch_z'])
    else:
        patch_size = None

    if args.dataset.lower() == 'nyumets':
        train_dict = get_nyumets_data(split='train', debug_subset=args.debug_subset)
        train_transforms = get_nyumets_transforms(
            split='train',
            temporal=use_temporal,
            patch_size=patch_size,
            intensity_augmentation=args.intensity_augmentation,
            spatial_augmentation=args.spatial_augmentation
        )
        val_dict = get_nyumets_data(split='val', debug_subset=args.debug_subset)
        val_transforms = get_nyumets_transforms(split='val')
    elif args.dataset.lower() == 'brats21':
        train_dict = get_brats21_data(split='train')
        train_transforms = get_brats21_transforms(
            split='train',
            patch_size=patch_size,
            intensity_augmentation=args.intensity_augmentation,
            spatial_augmentation=args.spatial_augmentation
        )
        val_dict = get_brats21_data(split='val')
        val_transforms = get_brats21_transforms(split='val')
    elif args.dataset.lower() == 'stanfordmets':
        train_dict = get_stanfordbrainmetsshare_data(split='train')
        train_transforms = get_stanfordbrainmetsshare_transforms(
            split='train',
            patch_size=patch_size,
            intensity_augmentation=args.intensity_augmentation,
            spatial_augmentation=args.spatial_augmentation
        )
        val_dict = get_stanfordbrainmetsshare_data(split='val')
        val_transforms = get_stanfordbrainmetsshare_transforms(split='val')
    else:
        raise NotImplementedError(f"Sorry, the dataset '{args.dataset}' is not supported.")

    if use_temporal:
        train_dataset = TemporalShuffleBuffer(
            train_dict,
            transform=train_transforms,
            buffer_size=args.buffer_size,
            combine_timepoints=use_temporal,
            sequence_limit=args.sequence_limit
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    else:
        train_dataset = CacheDataset(train_dict, transform=train_transforms, cache_rate=0.2, num_workers=1)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    if args.dataset.lower() == 'nyumets':
        val_dataset = TemporalIterableDataset(val_dict, transform=val_transforms, store_previous=True)
        calculate_longitudinal_metrics = True
    else:
        val_dataset = CacheDataset(val_dict, transform=val_transforms, cache_rate=0.2, num_workers=1)
        calculate_longitudinal_metrics = False

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1)
    

    loss_function = get_loss_function(loss_function=args.loss_function)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    # basic metrics
    binary_dice_metric = DiceMetric(include_background=False)
    hausdorff_95_metric = HausdorffDistanceMetric(include_background=False, percentile=95.)
    tumor_vol_metric = TumorVolume()

    # Runs connected components
    tumor_count_metric = TumorCount()
    small_tumor_count_metric = TumorCount(volume_threshold=config['small_tumor_vol_threshold'])

    # Requires connected components as inputs
    per_tumor_vol_metric = PerTumorVolume(is_onehot=config['is_onehot'])
    iou_per_class_metric = IoUPerClass(is_onehot=config['is_onehot'])
    fbeta_metric = FBeta(beta=config['beta'], is_onehot=config['is_onehot'])

    # Longitudinal metrics
    change_tumor_count_metric = ChangeTumorCount()
    change_small_tumor_count_metric = ChangeTumorCount(volume_threshold=config['small_tumor_vol_threshold'])
    change_tumor_vol_metric = ChangeTumorVolume()
    change_per_tumor_vol_metric = ChangePerTumorVolume(is_onehot=config['is_onehot'])
    
    # choose inferer
    if args.use_sliding_window_inferer:
        sliding_window_inferer_roi_size = (config['inferer_roi_x'], config['inferer_roi_y'], config['inferer_roi_z'])
        inferer = SlidingWindowInferer(roi_size=sliding_window_inferer_roi_size)
    else:
        inferer = SimpleInferer()

    for epoch in range(args.epochs):
        model.train()
        step = 0
        epoch_loss = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"],
                batch_data["label"],
            )

            if use_temporal:
                if args.spatial_augmentation:
                    inputs, labels = spatial_augment_batch(
                        inputs, labels,
                        flip_x_prob=0.1, flip_y_prob=0.1,
                        rotate_prob=0.1, max_rotate_radians=0.3
                    )

                if args.use_patches:
                    inputs, labels = patch_batch(inputs, labels, patch_size=patch_size)

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_dataset) // train_loader.batch_size}, "
               f"train_loss: {loss.item():.4f}")
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )

                    val_outputs = inferer(val_inputs, model)

                    val_outputs = val_outputs.cpu()
                    val_labels = val_labels.cpu()
                    val_inputs = val_inputs.cpu()

                    (preds, targets) = post_processing_transforms(val_outputs, val_labels)

                    # compute metric for current iteration
                    binary_dice_metric(y_pred=preds, y=targets)
                    hausdorff_95_metric(y_pred=preds, y=targets)
                    tumor_vol_metric(y_pred=preds, y=targets)

                    if args.calculate_extended_metrics or best_metric > 0.7:
                        tumor_count_metric(y_pred=preds, y=targets)
                        small_tumor_count_metric(y_pred=preds, y=targets)

                        (preds, targets) = cc_processing_transforms(val_outputs, val_labels)

                        iou_per_class_metric(y_pred=preds, y=targets)
                        per_tumor_vol_metric(y_pred=preds, y=targets)
                        fbeta_metric(y_pred=preds, y=targets)

                        if calculate_longitudinal_metrics:
                            val_prev_images, val_prev_labels = val_data['prev_image'], val_data['prev_label']
                            
                            (preds, targets, post_prev_labels) = longitudinal_transforms(
                                val_outputs, val_labels, val_inputs, val_prev_images, val_prev_labels)

                            change_tumor_count_metric(y_pred=preds, y=targets, y_prev=post_prev_labels)
                            change_small_tumor_count_metric(y_pred=preds, y=targets, y_prev=post_prev_labels)
                            change_tumor_vol_metric(y_pred=preds, y=targets, y_prev=post_prev_labels)
                            change_per_tumor_vol_metric(y_pred=preds, y=targets, y_prev=post_prev_labels)

                # aggregate the final mean dice result
                dice_agg = binary_dice_metric.aggregate().item()
                hd95_agg = hausdorff_95_metric.aggregate().item()
                tumor_vol_agg = tumor_vol_metric.aggregate().item()
                
                # reset the status for next validation round
                binary_dice_metric.reset()
                hausdorff_95_metric.reset()
                tumor_vol_metric.reset()
                
                print(
                    f"current epoch: {epoch + 1} current mean dice: {dice_agg:.4f}"
                    f"\ncurrent mean hausdorff distance (95%): {hd95_agg:.4f}"
                    f"\ncurrent MAE tumor volume: {tumor_vol_agg:.4f}"
                )


                if args_calculate_extended_metrics or best_metric > 0.7:
                    tumor_count_agg = tumor_count_metric.aggregate().item()  
                    small_tumor_count_agg = small_tumor_count_metric.aggregate().item()                               
                    per_tumor_vol_agg = per_tumor_vol_metric.aggregate().item()
                    iou_per_class_agg = iou_per_class_metric.aggregate().item()
                    fbeta_agg = fbeta_metric.aggregate().item()
                    
                    tumor_count_metric.reset()
                    small_tumor_count_metric.reset()
                    per_tumor_vol_metric.reset()
                    iou_per_class_metric.reset()
                    fbeta_metric.reset()
                    
                    print(
                        f"current MAE tumor count: {dice_agg:.4f}"
                        f"\ncurrent MAE small tumor count: {hd95_agg:.4f}"
                        f"\ncurrent MAE per tumor volume: {per_tumor_vol_agg:.4f}"
                        f"\ncurrent IOU per class: {iou_per_class_agg:.4f}"
                        f"\ncurrent FBeta: {fbeta_agg:.4f}"
                    )
                    
                    if calculate_longitudinal_metrics:
                        change_tumor_vol_agg = change_tumor_vol_metric.aggregate().item()
                        change_tumor_count_agg = change_tumor_count_metric.aggregate().item()
                        change_small_tumor_count_agg = change_small_tumor_count_metric.aggregate().item()
                        change_per_tumor_vol_agg = change_per_tumor_vol_metric.aggregate().item()

                        change_tumor_vol_metric.reset()
                        change_tumor_count_metric.reset()
                        change_small_tumor_count_metric.reset()
                        change_per_tumor_vol_metric.reset()
                        print(
                            f"\ncurrent MAE change tumor volume: {change_tumor_vol_agg:.4f}"
                            f"\ncurrent MAE change tumor count: {change_tumor_count_agg:.4f}"
                            f"\ncurrent MAE change small tumor count: {change_small_tumor_count_agg:.4f}"
                            f"\ncurrent MAE change per tumor volume: {change_per_tumor_vol_agg:.4f}"
                        )

                if dice_agg > best_metric:
                    best_metric = dice_agg
                    best_metric_epoch = epoch + 1

                print(
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
                
                if args.ckpt_dir is not None:
                    torch.save(model.state_dict(), os.path.join(
                    args.ckpt_dir, f"{args.dataset}_{args.model}-epoch-{str(epoch + 1)}.pth"))

                    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
