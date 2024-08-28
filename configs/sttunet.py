import torch.nn as nn

from optuna import Trial, Study
from .base import BaseModelConfig

from argparse import Namespace, ArgumentParser

from nyumets.networks.nets.sttunet import STTUNet


class STTUNetConfig(BaseModelConfig):
    model_name = "unet_stt"

    def setup_search_space(self, trial: Trial, args: Namespace):
        args.last_feature_size = trial.suggest_int("last_feature_size", 32, 64)
        return super().setup_search_space(trial, args)

    def load_best_config(self, study: Study, args: Namespace):
        best_params = study.best_params
        args.last_feature_size = best_params["last_feature_size"]
        return super().load_best_config(study, args)

    def add_model_specific_args(self, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("UNet_STT")
        parser.add_argument("--last_feature_size", type=int, default=32)
        return super().add_model_specific_args(parent_parser)

    def instantiate_model(self) -> nn.Module:
        args = self.get_parsed_args()
        return STTUNet(
            in_channels=args.in_channels,
            out_channels=2,
            features=(32, 32, 64, 128, 256, args.last_feature_size)
        )
