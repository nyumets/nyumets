from .base import BaseModelConfig

from optuna import Trial, Study

from argparse import ArgumentParser, Namespace


def _get_config_class(temp_args):
    registry = BaseModelConfig.__subclasses__()
    net_names = [net_type.model_name for net_type in registry]
    if temp_args.model.lower() not in net_names:
        raise NotImplementedError(f"Sorry, the model '{temp_args.model}' is not supported.")
    else:
        return registry[net_names.index(temp_args.model.lower())]


def get_model_config(temp_args: Namespace, parser: ArgumentParser) -> BaseModelConfig:
    return _get_config_class(temp_args)(parser, temp_args)

def get_model_config_from_trial(temp_args: Namespace, parser: ArgumentParser, trial: Trial) -> BaseModelConfig:
    config = get_model_config(temp_args, parser)
    config.setup_search_space(trial, temp_args)
    return config

def get_model_config_from_study(temp_args: Namespace, parser: ArgumentParser, study: Study) -> BaseModelConfig:
    config = get_model_config(temp_args, parser)
    config.load_best_config(study, temp_args)
    return config
