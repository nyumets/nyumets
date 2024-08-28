import yaml
from pathlib import Path

from nyumets.transforms.utils import patch_batch, spatial_augment_batch
from monai.inferers import SimpleInferer, SlidingWindowInferer

from nyumets.transforms.utils import (
    post_processing_transforms,
    cc_processing_transforms,
    longitudinal_transforms,
)
from nyumets.metrics.tumor import (
    TumorCount,
    TumorVolume,
    PerTumorVolume,
    ChangeTumorCount,
    ChangeTumorVolume,
    ChangePerTumorVolume,
    FBeta,
)
from nyumets.metrics.iou import IoUPerClass
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from torchmetrics import MaxMetric

import torch
import torch.nn as nn

from typing import Any
from pytorch_lightning import LightningModule

from nyumets.losses.utils import get_loss_function


with open(Path(__file__).parents[1] / "config.yaml") as f:
    config = yaml.safe_load(f)


class BaselineModule(LightningModule):
    @staticmethod
    def add_model_generic_args(parent_parser):
        parser = parent_parser.add_argument_group("Meta")
        parser.add_argument("--model", type=str, default=config['model'])
        parser.add_argument("--use_sliding_window_inferer", type=bool, default=config['use_sliding_window_inferer'])
        parser.add_argument(
            "--ckpt_dir",
            type=str,
            default=config['ckpt_dir'],
            help="Directory to save checkpoints. Defaults to None (will not save checkpoint)",
        )

        parser = parent_parser.add_argument_group("Training")
        parser.add_argument("--use_patches", type=int, default=config['use_patches'])

        parser = parent_parser.add_argument_group("Metrics")
        parser.add_argument(
            "--always_calculate_extended_metrics",
            type=bool,
            default=config['always_calculate_extended_metrics'],
        )

        parser.add_argument(
            "--extended_metrics_thres",
            type=float,
            default=100
            # TODO: Set to highest so we won't ever eval. We have mem frag now.
        )
        return parent_parser

    def __init__(self, _model: nn.Module, **kwargs: Any):
        super().__init__()
        self.net = _model
        self.save_hyperparameters(ignore="_model")

        # loss function
        self.loss_function = get_loss_function(self.hparams.loss_function)

        # Keep track of best val loss
        self.best_val_metric = MaxMetric()

        # Setup inferer
        if self.hparams.use_sliding_window_inferer:
            sliding_window_inferer_roi_size = (config['inferer_roi_x'], config['inferer_roi_y'], config['inferer_roi_z'])
            self.inferer = SlidingWindowInferer(roi_size=sliding_window_inferer_roi_size)
        else:
            self.inferer = SimpleInferer()

        # Setup metrics
        stages = ["train", "val", "test"]
        self.simple_metrics = {k: self._setup_simple_metric() for k in stages}

        stages = stages[1:]
        self.extended_metrics = {k: self._setup_extended_metric() for k in stages}

        # Book keeping for extended metrics
        self.did_val_extended_metrics = False

    @staticmethod
    def _setup_simple_metric():
        # basic metrics
        basic_ret = {}
        basic_ret["binary_dice_metric"] = DiceMetric(include_background=False)
        basic_ret["hausdorff_95_metric"] = HausdorffDistanceMetric(
            include_background=False,
            percentile=95.0,
        )
        basic_ret["tumor_vol_metric"] = TumorVolume()

        return basic_ret

    @staticmethod
    def _setup_extended_metric():
        # Runs connected components
        nocc_ret = {}
        nocc_ret["tumor_count_metric"] = TumorCount()
        nocc_ret["small_tumor_count_metric"] = TumorCount(
            volume_threshold=config["small_tumor_vol_threshold"],
        )

        # Requires connected components as inputs
        cc_ret = {}
        cc_ret["per_tumor_vol_metric"] = PerTumorVolume(
            is_onehot=config["is_onehot"],
        )
        cc_ret["iou_per_class_metric"] = IoUPerClass(
            is_onehot=config["is_onehot"],
        )
        cc_ret["fbeta_metric"] = FBeta(
            beta=config["beta"], is_onehot=config["is_onehot"]
        )

        # Longitudinal metrics
        lo_ret = {}
        lo_ret["change_tumor_count_metric"] = ChangeTumorCount()
        lo_ret["change_small_tumor_count_metric"] = ChangeTumorCount(
            volume_threshold=config["small_tumor_vol_threshold"],
        )
        lo_ret["change_tumor_vol_metric"] = ChangeTumorVolume()
        lo_ret["change_per_tumor_vol_metric"] = ChangePerTumorVolume(
            is_onehot=config["is_onehot"]
        )

        return nocc_ret, cc_ret, lo_ret

    def eval_mid_simple(self, preds, targets, stage):
        _preds, _targets = preds.cpu(), targets.cpu()
        (preds, targets) = post_processing_transforms(_preds, _targets)

        basic_out = {
            k: v(preds, targets) for k, v in self.simple_metrics[stage].items()
        }
        return basic_out

    def eval_mid_extended(
        self,
        preds,
        targets,
        inputs,
        prev_inputs,
        prev_labels,
        stage,
    ):
        ret = {}

        _preds, _targets = preds.cpu(), targets.cpu()
        (preds, targets) = post_processing_transforms(_preds, _targets)

        ret.update({k: v(preds, targets) for k, v in self.extended_metrics[stage][0].items()})

        (preds, targets) = cc_processing_transforms(_preds, _targets)
        ret.update({k: v(preds, targets) for k, v in self.extended_metrics[stage][1].items()})

        (preds, targets, _prev_labels) = longitudinal_transforms(
            _preds, _targets, inputs, prev_inputs, prev_labels
        )
        ret.update({
            k: v(preds, targets, _prev_labels)
            for k, v in self.extended_metrics[stage][2].items()
        })

        return ret

    def eval_end_extended(self, stage):
        # First aggregate all metrics
        out = {}
        for i in range(3):
            out.update({k: v.aggregate() for k, v in self.extended_metrics[stage][i].items()})

        # Then reset all metrics
        for i in range(3):
            {k: v.reset() for k, v in self.extended_metrics[stage][i].items()}

        return out

    def eval_end_simple(self, stage):
        # First aggregate all metrics
        out = {}
        out.update({k: v.aggregate() for k, v in self.simple_metrics[stage].items()})

        # Then reset all metrics
        {k: v.reset() for k, v in self.simple_metrics[stage].items()}

        return out

    def log_metrics(self, metrics, stage="train", suffix="", **kwargs):
        for k, v in metrics.items():
            v = v.mean()
            self.log(f"{stage}/{k}{suffix}", v, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any, use_inferer: bool = False):
        x, y = batch["image"], batch["label"]

        if self.hparams.use_temporal and self.training:
            if self.hparams.spatial_augmentation:
                x, y = spatial_augment_batch(
                    x, y,
                    flip_x_prob=0.1, flip_y_prob=0.1,
                    rotate_prob=0.1, max_rotate_radians=0.3
                )
            if self.hparams.use_patches:
                x, y = patch_batch(x, y)

        if use_inferer:
            out = self.inferer(x, self.net)
        else:
            out = self.forward(x)

        extra_loss = None
        if self.training and isinstance(out, tuple):
            # VAE returns tuple
            out, extra_loss = out

        if not extra_loss:
            extra_loss = 0

        loss = self.loss_function(out, y) + extra_loss

        return {"loss": loss, "preds": out, "targets": y}

    def do_metrics(self, loss, preds, targets, batch, stage="train"):
        # Log loss
        self.log(f"{stage}/loss", loss.mean())

        # Find simple metrics
        self.eval_mid_simple(preds, targets, stage)

        do_extended_metrics = False
        if stage == "val":
            m = self.best_val_metric.compute().mean()

            if self.hparams.always_calculate_extended_metrics or m > self.hparams.extended_metrics_thres:
                do_extended_metrics = True

                # Set the book keeping variable
                self.did_val_extended_metrics = True

        elif stage == "test":
            do_extended_metrics = True

        if do_extended_metrics:
            self.eval_mid_extended(
                preds,
                targets,
                batch["image"],
                batch["prev_image"],
                batch["prev_label"],
                stage=stage,
            )

    def on_train_epoch_start(self) -> None:
        self.best_val_metric.reset()
        super().on_train_epoch_start()

    def training_step(self, batch: Any, _):
        out = self.step(batch)
        self.do_metrics(
            out["loss"],
            out["preds"],
            out["targets"],
            batch,
            stage="train",
        )
        return out["loss"].mean()

    def training_epoch_end(self, _):
        m = self.eval_end_simple(stage="train")
        self.log_metrics(m, stage="train")

    def validation_step(self, batch: Any, _):
        out = self.step(batch, use_inferer=True)
        out.update({"batch": batch})
        self.do_metrics(
            out["loss"],
            out["preds"],
            out["targets"],
            batch,
            stage="val",
        )
        return out["loss"].mean()

    def validation_epoch_end(self, _):
        m = self.eval_end_simple(stage="val")
        self.log_metrics(m, stage="val")

        self.best_val_metric.update(m["binary_dice_metric"].mean())

        if self.did_val_extended_metrics:
            m = self.eval_end_extended(stage="val")
            self.log_metrics(m, stage="val")
            self.did_val_extended_metrics = False

    def test_step(self, batch: Any, _):
        out = self.step(batch, use_inferer=True)
        out.update({"batch": batch})
        self.do_metrics(
            out["loss"],
            out["preds"],
            out["targets"],
            batch,
            stage="test",
        )
        return out["loss"].mean()

    def test_epoch_end(self, _):
        m = self.eval_end_simple(stage="test")
        self.log_metrics(m, stage="test")

        m = self.eval_end_extended(stage="test")
        self.log_metrics(m, stage="test")

    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
            ),
        }
