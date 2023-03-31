from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError


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
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.L1Loss()

        # metric objects for calculating and averaging error across batches
        self.train_error = MeanAbsoluteError()
        self.val_error = MeanAbsoluteError()
        self.test_error = MeanAbsoluteError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation error
        self.val_error_best = MinMetric()

    def forward(self, x: torch.Tensor, noisy_x: torch.Tensor):
        return self.net(x, noisy_x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_error.reset()
        self.val_error_best.reset()

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
        x = batch
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
        return adj, x, denoised_x, loss

    def training_step(self, batch: Any, batch_idx: int):
        adj, x, denoised_x, loss = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_error(denoised_x, x)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/error", self.train_error, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        adj, x, denoised_x, loss = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_error(denoised_x, x)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/error", self.val_error, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        error = self.val_error.compute()  # get current val error
        self.val_error_best(error)  # update best so far val error
        # log `val_error_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/error_best", self.val_error_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        adj, x, denoised_x, loss = self.model_step(batch)
        # update and log metrics
        self.test_loss(loss)
        self.test_error(denoised_x, x)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/error", self.test_error, on_step=False, on_epoch=True, prog_bar=True)

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
