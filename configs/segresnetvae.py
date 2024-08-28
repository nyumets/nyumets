import torch.nn as nn
from argparse import Namespace, ArgumentParser

from .base import BaseModelConfig

from optuna import Trial
from monai.networks.nets import SegResNetVAE

class SegResNetVAEConfig(BaseModelConfig):
    model_name = "segresnetvae"

    def setup_search_space(self, trial: Trial, args: Namespace) -> Namespace:
        args.init_filters = trial.suggest_int("init_filters", 8, 32)
        return super().setup_search_space(trial, args)

    def add_model_specific_args(self, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("SegResNetVAE")
        parser.add_argument("--init_filters", type=int, default=8)
        return super().add_model_specific_args(parent_parser)

    def instantiate_model(self) -> nn.Module:
        args = self.get_parsed_args()
        return SegResNetVAE(
            input_image_size=self.image_size,
            init_filters=args.init_filters,
            out_channels=2,
            dropout_prob=args.dropout
        )
