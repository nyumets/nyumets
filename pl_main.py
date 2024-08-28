import torch
import optuna
from optuna import Trial

import wandb

from pathlib import Path
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from argparse import ArgumentParser

from nyumets.pl_train import BaselineModule
from nyumets.data.pl_datamodule import SharedDataModule

from pytorch_lightning import seed_everything

from configs import (
    get_model_config,
    get_model_config_from_trial,
    get_model_config_from_study
)

with open(Path(__file__).parent / "config.yaml") as f:
    config = yaml.safe_load(f)

# TODO: Needed to work around "received 0 items of ancdata" bug
torch.multiprocessing.set_sharing_strategy('file_system')


def main(local_config, **kwargs):
    # Attempt to reduce fragmentation
    torch.cuda.empty_cache()

    # Obtain args
    args = local_config.get_parsed_args()

    if "trial" in kwargs.keys():
        trial_id = kwargs["trial"].number
        study = kwargs["study_name"]
    else:
        trial_id = None
        study = None

    # If we are searching, only train for 10 epochs
    if study:
        args.max_epochs = 10

    # Create model container
    model = BaselineModule(_model=local_config.instantiate_model(), **vars(args))

    # Instantiate callbacks
    callbacks = [EarlyStopping(monitor="val/binary_dice_metric", patience=10, mode="max")]
    if args.ckpt_dir:
        ckpt_name = f"{args.model}-trial{trial_id}-seed{args.seed}"
        callbacks.append(
            ModelCheckpoint(
                filename=ckpt_name,
                dirpath=args.ckpt_dir,
                monitor="val/binary_dice_metric",
                mode="max",
                save_top_k=1,
            )
        )

    if config["use_wandb"]:
        # Instantiate logger
        logger = WandbLogger(
            project="NYUMets",
            entity="olab",
            tags=[kwargs["study_name"]] if study else None,
        )
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    else:
        logger = None

    # Create trainer
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
    )

    dm = SharedDataModule(**vars(args))

    # Start training
    trainer.fit(model, datamodule=dm)

    # Terminate wandb
    wandb.finish()

    if args.do_test:
        # Test
        trainer.test(datamodule=dm, ckpt_path="best")

    return model.best_val_metric.compute()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hparam_search", action="store_true")
    parser.add_argument("--seed", type=int, default=config["random_seed"])

    # Add model specific args
    parser = BaselineModule.add_model_generic_args(parser)

    # Add data specific args
    parser = SharedDataModule.add_dataset_specific_args(parser)

    # Add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    # Change some defaults
    parser.set_defaults(
        check_val_every_n_epoch=config["val_interval"],
        accelerator=config["accelerator"],
        strategy=config["strategy"] if "strategy" in config.keys() else None,
        devices=config["devices"],
        max_epochs=config["epochs"]
    )
    temp_args = parser.parse_known_args()[0]

    if temp_args.hparam_search:
        parser.add_argument("--num_trials", type=int, default=40)
        parser.add_argument("--no_load", action="store_true")
        parser.add_argument("--run_best", action="store_true")
        parser.add_argument("--study_name", type=str, default=None)

        temp_args = parser.parse_known_args()[0]

        if temp_args.study_name is None:
            study_name = f"study_{temp_args.model}"
        else:
            study_name = temp_args.study_name

        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///studies/{study_name}.db",
            study_name=study_name,
            load_if_exists=not temp_args.no_load,
        )

        if temp_args.run_best:
            # Set seeds
            seed_everything(temp_args.seed)

            model_config = get_model_config_from_study(temp_args, parser, study)
            main(model_config)
        else:
            # Define the objective function
            def objective(trial: Trial) -> torch.Tensor:
                model_config = get_model_config_from_trial(temp_args, parser, trial)
                return main(model_config, trial=trial, study_name=study_name)

            study.optimize(objective, n_trials=temp_args.num_trials)
    else:
        # Set seeds
        seed_everything(temp_args.seed)

        # Collect the model specific args
        local_config = get_model_config(temp_args, parser)
        main(local_config)
