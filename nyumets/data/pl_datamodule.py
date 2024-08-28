from pathlib import Path
import yaml

from pytorch_lightning import LightningDataModule
from monai.data import CacheDataset, DataLoader

from nyumets.data.utils import get_nyumets_data, get_brats21_data
from nyumets.transforms.utils import get_nyumets_transforms, get_brats21_transforms

from nyumets.data.dataset import TemporalShuffleBuffer, TemporalIterableDataset

with open(Path(__file__).parents[2] / "config.yaml") as f:
    config = yaml.safe_load(f)


class SharedDataModule(LightningDataModule):
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")
        parser.add_argument("--dataset", type=str, default=config['dataset'], help="Name of dataset for training. Options: 'nyumets', 'brats21'")
        parser.add_argument("--pin_memory", type=bool, default=False)
        parser.add_argument("--buffer_size", type=int, default=config['buffer_size'])
        parser.add_argument("--sequence_limit", type=int, default=config['sequence_limit'])
        parser.add_argument("--debug_subset", type=int, default=None)
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument("--image_modalities",  nargs='+', default=config['image_modalities'])
        parser.add_argument("--clean_label", action="store_true")

        # Dataset boilerplate
        parser.add_argument("--spatial_augmentation", type=bool, default=config['spatial_augmentation'])
        parser.add_argument("--intensity_augmentation", type=bool, default=config['intensity_augmentation'])

        parser.add_argument("--train_batch_size", type=int, default=32, help="The batch size for training set.")
        parser.add_argument("--eval_batch_size", type=int, default=32, help="The batch size for test and val set.")
        parser.add_argument("--train_num_workers", type=int, default=4, help="The number of workers for training set.")
        parser.add_argument("--eval_num_workers", type=int, default=4, help="The number of workers for test and val set.")
        return parent_parser

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.use_patches:
            patch_size = (config['patch_x'], config['patch_y'], config['patch_z'])
        else:
            patch_size = None

        if self.hparams.dataset.lower() == 'nyumets':
            train_dict = get_nyumets_data(
                split='train',
                image_modalities=self.hparams.image_modalities,
                debug_subset=self.hparams.debug_subset,
                expert_adjusted_only=self.hparams.clean_label
            )
            train_transforms = get_nyumets_transforms(
                image_modalities=self.hparams.image_modalities,
                temporal=self.hparams.use_temporal,
                patch_size=patch_size,
                intensity_augmentation=self.hparams.intensity_augmentation,
                spatial_augmentation=self.hparams.spatial_augmentation
            )
            val_dict = get_nyumets_data(
                split='val',
                image_modalities=self.hparams.image_modalities,
                debug_subset=self.hparams.debug_subset,
                expert_adjusted_only=self.hparams.clean_label
            )
            val_transforms = get_nyumets_transforms(
                image_modalities=self.hparams.image_modalities,
                patch_size=patch_size
            )
            if self.hparams.do_test:
                test_dict = get_nyumets_data(
                    split='test',
                    image_modalities=self.hparams.image_modalities,
                    debug_subset=self.hparams.debug_subset,
                    expert_adjusted_only=self.hparams.clean_label
                )
                test_transforms = get_nyumets_transforms(
                    image_modalities=self.hparams.image_modalities,
                    patch_size=patch_size
                )

        elif self.hparams.dataset.lower() == 'brats21':
            train_dict = get_brats21_data(split='train')
            train_transforms = get_brats21_transforms()
            val_dict = get_brats21_data(split='val')
            val_transforms = get_brats21_transforms()
            if self.hparams.do_test:
                test_dict = get_brats21_data(split='test')
                test_transforms = get_brats21_transforms()
        else:
            raise NotImplementedError(f"Sorry, the dataset '{self.hparams.dataset}' is not supported.")

        if self.hparams.use_temporal:
            self.train_dataset = TemporalShuffleBuffer(
                train_dict,
                transform=train_transforms,
                buffer_size=self.hparams.buffer_size,
                combine_timepoints=self.hparams.use_temporal,
                sequence_limit=self.hparams.sequence_limit
            )
        else:
            self.train_dataset = CacheDataset(
                train_dict,
                transform=train_transforms,
                cache_rate=0.2,
                num_workers=4,
            )
        self.val_dataset = TemporalIterableDataset(
            val_dict,
            transform=val_transforms,
            store_previous=True,
        )
        if self.hparams.do_test:
            self.test_dataset = TemporalIterableDataset(
                test_dict,
                transform=test_transforms,
                store_previous=True,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=1 if self.hparams.use_temporal else self.hparams.train_batch_size,
            num_workers=1 if self.hparams.use_temporal else self.hparams.train_num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=not self.hparams.use_temporal
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=1 if self.hparams.use_temporal else self.hparams.eval_batch_size,
            num_workers=1 if self.hparams.use_temporal else self.hparams.eval_num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.hparams.do_test:
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.hparams.eval_batch_size,
                num_workers=self.hparams.eval_num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
