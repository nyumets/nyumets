import torch.nn as nn

from optuna import Trial, Study

from typing import Optional, Sequence
from argparse import Namespace, ArgumentParser

import yaml
from pathlib import Path


with open(Path(__file__).parents[1] / "config.yaml") as f:
    config = yaml.safe_load(f)


class BaseModelConfig:
    image_size: Sequence[int] = (96, 96, 96)
    model_name: Optional[str] = None

    def __init__(self, parser: ArgumentParser, temp_args: Namespace):
        self.args = None
        self.temp_args = temp_args

        # TODO: Dangerous workaround. This breaks on second trial
        try:
            self.parser = self.add_model_specific_args(parser)
        except:
            pass

    def load_best_config(self, study: Study, args: Namespace) -> Namespace:
        best_params = study.best_params
        args.dropout = best_params["dropout"]
        args.lr = best_params["lr"]
        args.loss_function = best_params["loss_function"]
        args.spatial_augmentation = best_params["spatial_aug"]
        args.intensity_augmentation = best_params["intensity_aug"]
        args.in_channels = len(best_params['image_modalities'])

        self.args = args
        return args

    def setup_search_space(self, trial: Trial, args: Namespace) -> Namespace:
        args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
        args.lr = trial.suggest_float("lr", 1e-5, 1e-2)
        args.loss_function = trial.suggest_categorical(
            "loss_function",
            [
                "dice",
                "dice_focal",
                "focal",
                "generalized_dice",
                "generalized_dice_focal",
                "dice_ce"
            ]
        )
        args.spatial_augmentation = trial.suggest_categorical("spatial_aug", [True, False])
        args.intensity_augmentation = trial.suggest_categorical("intensity_aug", [True, False])
        args.in_channels = len(args.image_modalities)
        self.args = args
        return args

    def instantiate_model(self, args: Namespace) -> nn.Module:
        raise NotImplementedError()

    def add_model_specific_args(self, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Generic Model Hyperparameters")
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--loss_function", type=str, default=config['loss_function'])
        self.parser = parent_parser
        return parent_parser

    def get_parsed_args(self) -> Namespace:
        if self.args is None:
            self.args = self.parser.parse_args()

        self.args.use_temporal = self.args.model in ["unet_stt"]
        self.args.in_channels = len(self.args.image_modalities)

        return self.args
