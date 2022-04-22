import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data
from callback import AttackCallback


def main(args):

    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.logger == "wandb":
        logger = WandbLogger(project="protective perturbation mmsys2022", name=args.description)
    else:
        logger = TensorBoardLogger("protective perturbation mmsys2022", name=args.description)

    checkpoint = ModelCheckpoint(monitor="loss_val/total", mode="min", save_last=False)

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger if not bool(args.dev + args.test) else None,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        checkpoint_callback=checkpoint,
        precision=args.precision,
        callbacks=[AttackCallback()],
    )

    if args.model_A != args.model_B:
        from module import AttackModule
        model = AttackModule(args)
    else:
        from module_noModelB import AttackModule
        model = AttackModule(args)

    data = CIFAR10Data(args)

    if bool(args.test):
        trainer.test(model, data.test_dataloader())
    else:
        trainer.fit(model, data)
        trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--description", type=str, default="default")
    parser.add_argument("--data_dir", type=str, default="/home/tangbao/data/cifar10")
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
    )

    # MODULE args
    parser.add_argument(
        "--optimizer", type=str, default="Adam", choices=["SGD", "Adam"]
    )
    parser.add_argument("--learning_rate", type=float, default=5e-3)  # 1e-3 for googlenet-vgg13bn, 5e-3 for other models
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--model_A", type=str, default="vgg16_bn")
    parser.add_argument("--model_B", type=str, default="resnet18")
    parser.add_argument("--ssim_weight", type=float, default=0.5)

    # TRAINER args
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()

    main(args)
