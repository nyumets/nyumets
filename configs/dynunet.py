import torch.nn as nn
from argparse import Namespace, ArgumentParser

from .base import BaseModelConfig

from optuna import Trial
from monai.networks.nets import DynUNet


class DynUNetConfig(BaseModelConfig):
    model_name = "dynunet"

    def setup_search_space(self, trial: Trial, args: Namespace) -> Namespace:
        args.num_layers = trial.suggest_int("num_layers", 4, 5)
        return super().setup_search_space(trial, args)

    def add_model_specific_args(self, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("DynUNet")
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--conv_stride", type=int, default=2)
        return super().add_model_specific_args(parent_parser)

    def instantiate_model(self) -> nn.Module:
        args = self.get_parsed_args()
        strides = [1]+[2] * (args.num_layers-1)
        return DynUNet(
            spatial_dims=3,
            kernel_size=[3] * args.num_layers,
            strides=strides,
            upsample_kernel_size=strides[1:],
            in_channels=args.in_channels,
            out_channels=2,
            dropout=args.dropout,
        )
