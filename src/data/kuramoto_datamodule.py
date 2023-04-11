import os
from os import path as osp
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from kuramoto import Kuramoto
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


class KuramotoDataModule(LightningDataModule):
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
    - seed

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
        n_clusters: int = 2,
        cluster_size: int = 4,
        n_timesteps: int = 300,
        n_train: int = 300,
        n_val: int = 100,
        n_test: int = 100,
        seed: int = 0,  # Maybe possible to set seed globally elsewhere?
        batch_size: int = 64,
        num_workers: int = 0,
        data_dir: str = "data/kuramoto",
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.prefix = "{seed}_{n_clusters}_{cluster_size}".format(
            seed=self.hparams.seed,
            n_clusters=self.hparams.n_clusters,
            cluster_size=self.hparams.cluster_size,
        )
        self.dataset_path = osp.join(self.hparams.data_dir, self.prefix)

        self.original_adj: torch.tensor = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def create_adjacency_matrix(self):
        cluster_blocks = []
        class_blocks = []
        for n in range(0, self.hparams.n_clusters):
            cluster_blocks.append(
                torch.ones([self.hparams.cluster_size, self.hparams.cluster_size])
            )
            class_blocks.append(torch.ones(self.hparams.cluster_size, dtype=torch.int64) * n)
        single_sample_adj = torch.block_diag(*cluster_blocks)

        return single_sample_adj

    def create_samples(self):
        single_sample_adj = self.create_adjacency_matrix()
        nat_freqs = np.random.normal(
            loc=10, scale=1.0, size=self.hparams.n_clusters * self.hparams.cluster_size
        )
        model = Kuramoto(
            coupling=10,
            dt=0.001,
            T=self.hparams.n_timesteps
            * 2
            * 0.001,  # double timesteos to cover train and val/test ranges
            n_nodes=len(single_sample_adj.numpy()),
            natfreqs=nat_freqs,
        )

        act_mats_train = []
        for i in tqdm(range(0, self.hparams.n_train)):
            act_mat = model.run(
                adj_mat=single_sample_adj.numpy(),
                angles_vec=np.random.uniform(
                    0, 2 * np.pi, size=self.hparams.n_clusters * self.hparams.cluster_size
                ),
            )
            act_mats_train.append(act_mat)
        act_mats_train_nd = np.vstack(act_mats_train)
        act_mats_train_t = torch.from_numpy(act_mats_train_nd).type(torch.FloatTensor)[
            :, 0 : self.hparams.n_timesteps
        ]

        act_mats_val = []
        for i in tqdm(range(0, self.hparams.n_val)):
            act_mat = model.run(
                adj_mat=single_sample_adj.numpy(),
                angles_vec=np.random.uniform(
                    0, 2 * np.pi, size=self.hparams.n_clusters * self.hparams.cluster_size
                ),
            )
            act_mats_val.append(act_mat)
        act_mats_val_nd = np.vstack(act_mats_val)
        act_mats_val_t = torch.from_numpy(act_mats_val_nd).type(torch.FloatTensor)[
            :, self.hparams.n_timesteps : self.hparams.n_timesteps * 2
        ]

        # Shuffling rows of the adjacency matrix
        r = torch.tensor([7, 1, 2, 3, 4, 5, 6, 0])
        c = torch.tensor([7, 1, 2, 3, 4, 5, 6, 0])
        shuffled_sample_adj = single_sample_adj[r[:, None], c]  # shuffles rows
        shuffled_sample_adj = single_sample_adj[r][:, c]  # shuffles columns

        act_mats_test = []
        for i in tqdm(range(0, self.hparams.n_test)):
            act_mat = model.run(
                adj_mat=shuffled_sample_adj.numpy(),
                angles_vec=np.random.uniform(
                    0, 2 * np.pi, size=self.hparams.n_clusters * self.hparams.cluster_size
                ),
            )
            act_mats_test.append(act_mat)
        act_mats_test_nd = np.vstack(act_mats_test)
        act_mats_test_t = torch.from_numpy(act_mats_test_nd).type(torch.FloatTensor)[
            :, self.hparams.n_timesteps : self.hparams.n_timesteps * 2
        ]

        return single_sample_adj, act_mats_train_t, act_mats_val_t, act_mats_test_t

    def save_data(self, single_sample_adj, features_train, features_val, features_test):
        # features
        torch.save(
            torch.sin(features_train).clone(), osp.join(self.dataset_path, "features_train.pt")
        )
        torch.save(torch.sin(features_val).clone(), osp.join(self.dataset_path, "features_val.pt"))
        torch.save(
            torch.sin(features_test).clone(), osp.join(self.dataset_path, "features_test.pt")
        )
        # adj_original
        torch.save(single_sample_adj, osp.join(self.dataset_path, "original_adj.pt"))

    def prepare_data(self):
        """Generate the dataset if it does not exist."""
        if not osp.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            single_sample_adj, features_train, features_val, features_test = self.create_samples()
            self.save_data(single_sample_adj, features_train, features_val, features_test)

    def setup(self, stage: Optional[str] = None):
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`. This method is called
        by lightning with both `trainer.fit()` and `trainer.test()`.
        """
        # load original adjacency matrix
        if self.original_adj is None:
            self.original_adj = torch.load(osp.join(self.dataset_path, "original_adj.pt"))

        # load datasets only if not loaded already
        if self.data_train is None:
            train_features = torch.load(osp.join(self.dataset_path, "features_train.pt")).reshape(
                -1, self.hparams.cluster_size * self.hparams.n_clusters, self.hparams.n_timesteps
            )
            self.data_train = TensorDataset(
                train_features, self.original_adj.repeat(train_features.shape[0], 1, 1)
            )
        if self.data_val is None:
            val_features = torch.load(osp.join(self.dataset_path, "features_val.pt")).reshape(
                -1, self.hparams.cluster_size * self.hparams.n_clusters, self.hparams.n_timesteps
            )
            self.data_val = TensorDataset(
                val_features, self.original_adj.repeat(val_features.shape[0], 1, 1)
            )
        if self.data_test is None:
            test_features = torch.load(osp.join(self.dataset_path, "features_test.pt")).reshape(
                -1, self.hparams.cluster_size * self.hparams.n_clusters, self.hparams.n_timesteps
            )
            self.data_test = TensorDataset(
                test_features, self.original_adj.repeat(test_features.shape[0], 1, 1)
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
    dm = KuramotoDataModule(seed=42)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    batch_features, batch_adjs = next(iter(train_loader))
    print(batch_features.shape, batch_adjs.shape)
