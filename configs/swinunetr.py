import torch.nn as nn

from optuna import Trial
from argparse import Namespace, ArgumentParser

from .base import BaseModelConfig

from monai.networks.nets import SwinUNETR

class SwinUNETRConfig(BaseModelConfig):
    model_name = "swinunetr"

    def setup_search_space(self, trial: Trial, args: Namespace) -> Namespace:
        args.depth_12 = trial.suggest_int("depth_12", 2, 3)
        args.depth_34 = trial.suggest_int("depth_34", 2, 3)
        return super().setup_search_space(trial, args)

    def add_model_specific_args(self, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("SwinUNETR")
        parser.add_argument("--feature_size", type=int, default=24)
        parser.add_argument("--depth_12", type=int, default=2)
        parser.add_argument("--depth_34", type=int, default=2)
        return super().add_model_specific_args(parent_parser)

    def instantiate_model(self) -> nn.Module:
        args = self.get_parsed_args()
        return SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=args.in_channels,
            out_channels=2,
            depths=[2]*2+[2]*2,
            feature_size=24,
            drop_rate=args.dropout,
        )
