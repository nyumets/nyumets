import torch.nn as nn
from argparse import Namespace, ArgumentParser

from .base import BaseModelConfig

from optuna import Trial
from monai.networks.nets import UNet
from monai.networks.layers import Norm

class UNetConfig(BaseModelConfig):
    model_name = "unet"

    def setup_search_space(self, trial: Trial, args: Namespace) -> Namespace:
        args.initial_features = trial.suggest_int("initial_features", 8, 16)
        args.num_layers = trial.suggest_int("num_layers", 3, 4)
        args.expansion = 2
        args.num_res_units = trial.suggest_int("num_res_units", 2, 4)
        return super().setup_search_space(trial, args)

    def add_model_specific_args(self, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("UNet")
        parser.add_argument("--initial_features", type=int, default=16)
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--expansion", type=int, default=2)
        parser.add_argument("--num_res_units", type=int, default=2)
        return super().add_model_specific_args(parent_parser)

    def instantiate_model(self) -> nn.Module:
        args = self.get_parsed_args()
        return UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=2,
            channels=[args.initial_features*args.expansion**i for i in range(args.num_layers)],
            strides=[2]*(args.num_layers-1),
            num_res_units=args.num_res_units,
            dropout=args.dropout,
            norm=Norm.BATCH
        )
