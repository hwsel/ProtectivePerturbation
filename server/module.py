from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.metrics import Accuracy
from torchvision.utils import make_grid

from unet import UNet
from utils.all_classifiers import all_classifiers
from utils.scheduler import WarmupCosineLR
from utils.ssim import SSIM
from utils.utils import unnormalize


class AttackModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model_A = all_classifiers[self.hparams.model_A](pretrained=True).eval()
        self.model_B = all_classifiers[self.hparams.model_B](pretrained=True).eval()

        self.attack_model = UNet()

        self.xcent_loss = torch.nn.CrossEntropyLoss()
        self.ssim_loss = SSIM(data_range=1, size_average=True)
        self.accuracy = Accuracy()

    def forward(self, x):
        x_adv = self.attack_model(x)
        self.model_A.eval()
        self.model_B.eval()

        modelA_pred = self.model_A(x_adv)
        modelB_pred = self.model_B(x_adv)
        return x_adv, modelA_pred, modelB_pred

    def training_step(self, batch, batch_nb):
        x, label = batch
        x_adv, modelA_pred, modelB_pred = self.forward(x)

        accA = 100 * self.accuracy(modelA_pred.max(1)[1], label)
        accB = 100 * self.accuracy(modelB_pred.max(1)[1], label)

        xcent_lossA = self.xcent_loss(modelA_pred, label)
        xcent_lossB = self.xcent_loss(modelB_pred, label)

        ssim_loss = self.ssim_loss(unnormalize(x), unnormalize(x_adv))

        # (gradient decent on A, gradient accent on B)
        loss_total = xcent_lossA - xcent_lossB + self.hparams.ssim_weight * ssim_loss

        self.log("accuracy_train/modelA", accA)
        self.log("accuracy_train/modelB", accB)
        self.log("loss_train/modelA", xcent_lossA)
        self.log("loss_train/modelB", xcent_lossB)
        self.log("loss_train/ssim", ssim_loss)
        self.log("loss_train/total", loss_total)
        return loss_total

    def validation_step(self, batch, batch_nb):
        x, label = batch
        x_adv, modelA_pred, modelB_pred = self.forward(x)

        accA = 100 * self.accuracy(modelA_pred.max(1)[1], label)
        accB = 100 * self.accuracy(modelB_pred.max(1)[1], label)

        xcent_lossA = self.xcent_loss(modelA_pred, label)
        xcent_lossB = self.xcent_loss(modelB_pred, label)

        ssim_loss = self.ssim_loss(unnormalize(x), unnormalize(x_adv))

        # (gradient decent on A, gradient accent on B)
        loss_total = xcent_lossA - xcent_lossB + self.hparams.ssim_weight * ssim_loss

        self.log("accuracy_val/modelA", accA)
        self.log("accuracy_val/modelB", accB)
        self.log("loss_val/modelA", xcent_lossA)
        self.log("loss_val/modelB", xcent_lossB)
        self.log("loss_val/ssim", ssim_loss)
        self.log("loss_val/total", loss_total)

    def test_step(self, batch, batch_nb):
        x, label = batch
        x_adv, modelA_pred, modelB_pred = self.forward(x)

        self.model_A.eval()
        self.model_B.eval()

        modelA_pred_groundTruth = self.model_A(x)
        modelB_pred_groundTruth = self.model_B(x)

        accA_groundTruth = 100 * self.accuracy(modelA_pred_groundTruth.max(1)[1], label)
        accB_groundTruth = 100 * self.accuracy(modelB_pred_groundTruth.max(1)[1], label)

        accA = 100 * self.accuracy(modelA_pred.max(1)[1], label)
        accB = 100 * self.accuracy(modelB_pred.max(1)[1], label)

        xcent_lossA = self.xcent_loss(modelA_pred, label)
        xcent_lossB = self.xcent_loss(modelB_pred, label)

        ssim_loss = self.ssim_loss(unnormalize(x), unnormalize(x_adv))

        # (gradient decent on A, gradient accent on B)
        loss_total = xcent_lossA - xcent_lossB + self.hparams.ssim_weight * ssim_loss

        self.log("accuracy_test/modelA", accA)
        self.log("accuracy_test/modelB", accB)
        self.log("accuracy_test/modelA_groundTruth", accA_groundTruth)
        self.log("accuracy_test/modelB_groundTruth", accB_groundTruth)
        self.log("loss_test/modelA", xcent_lossA)
        self.log("loss_test/modelB", xcent_lossB)
        self.log("loss_test/ssim", ssim_loss)
        self.log("loss_test/total", loss_total)

        if self.hparams.logger == "wandb":
            img_grid = [
                wandb.Image(
                    make_grid(x_adv[: 10 * 5], nrow=10),
                    caption="Test",
                )
            ]
            self.logger.experiment.log({"Adversarial_Image": img_grid})

            img_grid_orig = [
                wandb.Image(
                    make_grid(x[: 10 * 5], nrow=10),
                    caption="Test",
                )
            ]
            self.logger.experiment.log({"Original_Image": img_grid_orig})

    def configure_optimizers(self):
        if self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.attack_model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.attack_model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
