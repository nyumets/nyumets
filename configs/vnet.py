import torch.nn as nn
from argparse import Namespace, ArgumentParser

from .base import BaseModelConfig

from optuna import Trial
from monai.networks.nets import VNet

class VNetConfig(BaseModelConfig):
    model_name = "vnet"

    def setup_search_space(self, trial: Trial, args: Namespace) -> Namespace:
        return super().setup_search_space(trial, args)

    def add_model_specific_args(self, parent_parser: ArgumentParser) -> ArgumentParser:
        return super().add_model_specific_args(parent_parser)

    def instantiate_model(self) -> nn.Module:
        args = self.get_parsed_args()
        return VNet(out_channels=2, dropout_prob=args.dropout)
