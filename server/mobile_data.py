import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from PIL import Image
import os


class C10IMG_MOBILE(Dataset):
    def __init__(self, transform, data_dir):
        self.transform = transform
        self.data_dir = data_dir

    def __len__(self):
        return 5000

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.data_dir, str(item)+'.png')).convert('RGB')
        return self.transform(img)


class C10IMGDATA_MOBILE(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = C10IMG_MOBILE(transform=transform,
                                data_dir=self.hparams.data_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader


class C10IMG(Dataset):
    def __init__(self, transform, model_name, data_dir):
        self.transform = transform
        self.model_name = model_name
        self.data_dir = data_dir
        self.original_path = os.path.join(self.data_dir, 'cifar10')
        self.perturbed_path = os.path.join(self.data_dir, self.model_name)
        self.data_list = os.listdir(self.original_path)

    def __len__(self):
        return 5000

    def __getitem__(self, index):
        img_name = self.data_list[index]
        original_img = Image.open(os.path.join(self.original_path, img_name)).convert('RGB')
        perturbed_img = Image.open(os.path.join(self.perturbed_path, img_name))
        return self.transform(original_img), self.transform(perturbed_img), int(img_name[-5:-4])


class C10IMGDATA(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = C10IMG(transform=transform,
                         model_name=self.hparams.model_name,
                         data_dir=self.hparams.data_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader
