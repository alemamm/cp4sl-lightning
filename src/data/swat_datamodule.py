import os
from os import path as osp
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


class SWaTDataModule(LightningDataModule):
    """LightningDataModule for the synthetic Kuramoto dataset.

    The module is initialized with the parameters that define the dataset.
    If the dataset does not exist, it will be created. Otherwise a dataloader containing the dataset can readily be used.

    The dataset parameters include:
    - n_clusters
    - cluster_size
    - n_timesteps
    - n_train
    - n_val
    - n_test

    The DataModule implements 4 key methods:
        def prepare_data(self):
            # generate data, save to disk, etc...
            # It uses the parameters passed to the constructor to generate the dataset.
        def train_dataloader(self):
            # return train dataloader corresponding to the parameters passed to the constructor.
        def val_dataloader(self):
            # return validation dataloader corresponding to the parameters passed to the constructor.
        def test_dataloader(self):
            # return test dataloader corresponding to the parameters passed to the constructor.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        n_train: int = 49000,
        n_val: int = 290,  # Note that for SWaT, we use val only to monitor training
        n_test: int = 200,  # Note that for SWaT, we use val only to monitor final results
        batch_size: int = 32,
        num_workers: int = 0,
        data_dir: str = "data",
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset_path = osp.join(self.hparams.data_dir, "SWaT")

        self.normal_adj: torch.tensor = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`. This method is called
        by lightning with both `trainer.fit()` and `trainer.test()`.
        """
        # load original adjacency matrix
        if self.normal_adj is None:
            self.normal_adj = torch.load(osp.join(self.dataset_path, "normal_adj.pt"))

        features = torch.load(osp.join(self.dataset_path, "swat_snapshots.pt"))

        # load datasets only if not loaded already
        if self.data_train is None:
            train_features = features[: self.hparams.n_train]
            self.data_train = TensorDataset(
                train_features, self.normal_adj.repeat(train_features.shape[0], 1, 1)
            )
        if self.data_val is None:
            val_features = features[
                self.hparams.n_train : self.hparams.n_train + self.hparams.n_val
            ]
            self.data_val = TensorDataset(
                val_features, self.normal_adj.repeat(val_features.shape[0], 1, 1)
            )
        if self.data_test is None:
            test_features = features[-self.hparams.n_test :]
            self.data_test = TensorDataset(
                test_features, self.normal_adj.repeat(test_features.shape[0], 1, 1)
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    dm = SWaTDataModule()
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    batch_features, batch_adjs = next(iter(train_loader))
    print(batch_features.shape, batch_adjs.shape)
