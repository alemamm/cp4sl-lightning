from time import time
from typing import Any

import cv2 as cv
import numpy as np
import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from .components.utils import get_off_diagonal_elements, tril_values


class CP4SLLitModule(LightningModule):
    """LightningModule for structure learning via denoising.

    A LightningModule organizes PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        ratio: float,
        nr: int,
        noise: str,
        gen_type: str = "dynamic",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()  # torch.nn.L1Loss()

        # metric objects for calculating reconstruction error and averaging error across batches
        self.train_recon_error = MeanAbsoluteError()
        self.val_recon_error = MeanAbsoluteError()
        self.test_recon_error = MeanAbsoluteError()

        # metric objects for calculating reconstruction error and averaging error across batches
        self.train_graph_error = MeanAbsoluteError()
        self.val_graph_error = MeanAbsoluteError()
        self.test_graph_error = MeanAbsoluteError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation errors
        self.val_recon_error_best = MinMetric()
        self.val_graph_error_best = MinMetric()

    def forward(self, x: torch.Tensor, noisy_x: torch.Tensor):
        return self.net(x, noisy_x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_recon_error.reset()
        self.val_recon_error_best.reset()
        self.val_graph_error_best.reset()

    def get_random_mask(self, x):
        nones = torch.sum(x > 0.0).float()
        nzeros = x.shape[0] * x.shape[1] - nones
        pzeros = nones / nzeros / self.hparams.ratio * self.hparams.nr
        probs = torch.zeros(x.shape)
        probs[x == 0.0] = pzeros
        probs[x > 0.0] = 1 / self.hparams.ratio
        mask = torch.bernoulli(probs)
        return mask

    def model_step(self, batch: Any):
        x, original_adj = batch
        mask = self.get_random_mask(x)
        # apply noise
        if self.hparams.noise == "mask":
            noisy_x = x * (1 - mask)
        elif self.hparams.noise == "normal":
            noise = torch.normal(0.0, 0.1, size=x.shape)
            noisy_x = x + (noise * mask)
        denoised_x, adj = self.forward(x, noisy_x)
        indices = mask > 0
        loss = self.criterion(denoised_x[indices], x[indices])
        return original_adj, adj, x, denoised_x, loss

    def training_step(self, batch: Any, batch_idx: int):
        original_adj, adj, x, denoised_x, loss = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_recon_error(denoised_x, x)
        # scale adj to range [0, 1] for graph error calculation

        # not using batch of adjacency matrices if FP graph generation is used
        if len(adj.shape) < 3:
            original_adj = original_adj[0]

        # Calculate graph error ignoring diagonal and using normalized values
        self.train_graph_error(
            torch.nn.functional.normalize(get_off_diagonal_elements(adj), dim=len(adj.shape) - 2),
            torch.nn.functional.normalize(
                get_off_diagonal_elements(original_adj), dim=len(adj.shape) - 2
            ),
        )

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/recon_error",
            self.train_recon_error,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/graph_error",
            self.train_graph_error,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        original_adj, adj, x, denoised_x, loss = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_recon_error(denoised_x, x)

        full_adj = adj

        adj = get_off_diagonal_elements(adj)
        original_adj = get_off_diagonal_elements(original_adj)

        # scale adj to range [0, 1] for graph error calculation
        adj = (adj - adj.min()) / (adj.max() - adj.min())

        # not using batch of adjacency matrices if FP graph generation is used
        if len(adj.shape) < 3:
            original_adj = original_adj[0]

        self.val_graph_error(adj, original_adj)

        if self.global_step % 100 == 0:
            plt.clf()
            if len(full_adj.shape) < 3:
                plt.imshow(full_adj.detach().numpy())
            else:
                plt.imshow(np.median(full_adj.detach().numpy(), axis=0))
            plt.savefig("plots/" + str(time()) + "_val.png")

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/recon_error", self.val_recon_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/graph_error", self.val_graph_error, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_validation_epoch_end(self):
        recon_error = self.val_recon_error.compute()  # get current val reconstruction error
        self.val_recon_error_best(recon_error)  # update best so far val reconstruction error
        graph_error = self.val_graph_error.compute()  # get current val graph error
        self.val_graph_error_best(graph_error)  # update best so far val graph error
        # log `val_error_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/recon_error_best", self.val_recon_error_best.compute(), prog_bar=True)
        self.log("val/graph_error_best", self.val_graph_error_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        original_adj, adj, x, denoised_x, loss = self.model_step(batch)
        # update and log metrics
        self.test_loss(loss)
        self.test_recon_error(denoised_x, x)

        full_adj = adj

        adj = get_off_diagonal_elements(adj)
        original_adj = get_off_diagonal_elements(original_adj)

        # scale adj to range [0, 1] for graph error calculation
        adj = (adj - adj.min()) / (adj.max() - adj.min())

        # not using batch of adjacency matrices if FP graph generation is used
        if len(adj.shape) < 3:
            original_adj = original_adj[0]

        self.test_graph_error(adj, original_adj)

        plt.clf()
        if len(full_adj.shape) < 3:
            plt.imshow(full_adj.detach().numpy())
        else:
            plt.imshow(np.median(full_adj.detach().numpy(), axis=0))
        plt.savefig("plots/" + str(time()) + "_test.png")

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/recon_error", self.test_recon_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/graph_error", self.test_graph_error, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
