import torch.nn as nn
from argparse import Namespace, ArgumentParser

from .base import BaseModelConfig

from optuna import Trial
from monai.networks.nets import UNETR

class UNETRConfig(BaseModelConfig):
    model_name = "unetr"

    def setup_search_space(self, trial: Trial, args: Namespace) -> Namespace:
        args.feature_size = trial.suggest_int("feature_size", 8, 20)
        args.hidden_size = 2**trial.suggest_int("hidden_size_exponent", 6, 9)
        args.mlp_size = 2**trial.suggest_int("mlp_size_exponent", 6, 11)
        args.num_heads = 2**trial.suggest_int("num_heads_exponent", 0, 6)
        return super().setup_search_space(trial, args)

    def add_model_specific_args(self, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("UNETR")
        parser.add_argument("--feature_size", type=int, default=16)
        parser.add_argument("--hidden_size", type=int, default=512)
        parser.add_argument("--mlp_size", type=int, default=2048)
        parser.add_argument("--num_heads", type=int, default=16)
        return super().add_model_specific_args(parent_parser)

    def instantiate_model(self) -> nn.Module:
        args = self.get_parsed_args()
        return UNETR(
            in_channels=args.in_channels,
            out_channels=2,
            img_size=self.image_size,
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_size,
            num_heads=args.num_heads,
            dropout_rate=args.dropout,
        )
