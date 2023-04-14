from time import time
from typing import Any

import cv2 as cv
import numpy as np
import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from .components.utils import get_off_diagonal_elements


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
        noise: str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()  # torch.nn.L1Loss()

        # metric objects for calculating reconstruction error and averaging error across batches
        self.train_recon_error = MeanSquaredError()
        self.val_recon_error = MeanSquaredError()
        self.test_recon_error = MeanSquaredError()

        # metric objects for calculating reconstruction error and averaging error across batches
        self.train_adj_error = MeanAbsoluteError()
        self.val_adj_error = MeanAbsoluteError()
        self.test_adj_error = MeanAbsoluteError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation errors
        self.val_recon_error_best = MinMetric()
        self.val_adj_error_best = MinMetric()

    def forward(self, x: torch.Tensor, noisy_x: torch.Tensor):
        return self.net(x, noisy_x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_recon_error.reset()
        self.val_recon_error_best.reset()
        self.val_adj_error_best.reset()

    def model_step(self, batch: Any):
        x, original_adj = batch
        mask = torch.bernoulli(torch.ones_like(x) * 0.1)
        # apply noise
        if self.hparams.noise == "mask":
            noisy_x = x
            noisy_x[mask > 0] = 0.0
        elif self.hparams.noise == "normal":
            noise = torch.normal(0.0, 1.0, size=x.shape)
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

        adj = get_off_diagonal_elements(adj)
        original_adj = get_off_diagonal_elements(original_adj)

        # scale adj to range [0, 1] for graph error calculation
        adj = (adj - adj.min()) / (adj.max() - adj.min())

        # not using batch of adjacency matrices if FP graph generation is used
        if len(adj.shape) < 3:
            original_adj = original_adj[0]

        self.train_adj_error(adj, original_adj)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/rec_error",
            self.train_recon_error,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/adj_error",
            self.train_adj_error,
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

        self.val_adj_error(adj, original_adj)

        if self.current_epoch % 5 == 0:
            plt.clf()
            if len(full_adj.shape) < 3:
                plt.imshow(full_adj.detach().numpy())
            else:
                plt.imshow(np.median(full_adj.detach().numpy(), axis=0))
            plt.savefig("plots/" + str(time()) + "_" + str(self.current_epoch) + "_val.png")

            plt.clf()
            color = cm.rainbow(np.linspace(0, 1, adj.shape[1]))
            colormap = plt.cm.gist_rainbow
            colors = [colormap(i) for i in np.linspace(0, 1, adj.shape[1])]
            for i, color in enumerate(colors):
                random_sample = np.random.randint(0, x.shape[0])
                plt.plot(
                    x[random_sample, i, :].cpu().detach().numpy(),
                    color=color,
                    linestyle="solid",
                    alpha=0.7,
                )
                plt.plot(
                    denoised_x[random_sample, i, :].cpu().detach().numpy(),
                    color=color,
                    linestyle="dashed",
                    alpha=0.7,
                )
            plt.savefig(
                "plots/" + str(time()) + "_" + str(self.current_epoch) + "_x_denoised_x.png"
            )

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/rec_error", self.val_recon_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/adj_error", self.val_adj_error, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        recon_error = self.val_recon_error.compute()  # get current val reconstruction error
        self.val_recon_error_best(recon_error)  # update best so far val reconstruction error
        adj_error = self.val_adj_error.compute()  # get current val graph error
        self.val_adj_error_best(adj_error)  # update best so far val graph error
        # log `val_error_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/rec_error_best", self.val_recon_error_best.compute(), prog_bar=True)
        self.log("val/adj_error_best", self.val_adj_error_best.compute(), prog_bar=True)

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

        self.test_adj_error(adj, original_adj)

        plt.clf()
        if len(full_adj.shape) < 3:
            plt.imshow(full_adj.detach().numpy())
        else:
            plt.imshow(np.median(full_adj.detach().numpy(), axis=0))
        plt.savefig("plots/" + str(time()) + "_" + str(self.current_epoch) + "_test.png")

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/rec_error", self.test_recon_error, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/adj_error", self.test_adj_error, on_step=False, on_epoch=True, prog_bar=True
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
