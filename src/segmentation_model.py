import argparse
import logging
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import CenterCrop

from constants import SEGMENTATION_DATASET_DIR

MAX_CPUS = 8
GRADIENT_CLIP_VALUE = 0.1


class FileDataset(Dataset):
    """
    Extended from the torch.utils.data.Dataset, this class can be used in combination
    with a Dataloader to lazily load the dataset using files as examples.
    """

    def __init__(self, dataset_dir, example_names):
        """
        :param dataset_dir: str, path to the folder with all examples
        :param example_names: list of str with all example names
        """
        self.dataset_dir = dataset_dir
        self.example_names = example_names
        self.transform = CenterCrop((224, 224))

    def __len__(self):
        """
        :returns: the total number of examples in the dataset
        """
        return len(self.example_names)

    def __getitem__(self, idx):
        """
        Given an index, queries the database to retrieve data for all sensors

        :param idx: int, specifying which record to load
        :returns: encoder input, decoder input, targets as tensors
        """
        input_tensor, target_tensor = torch.load(
            os.path.join(self.dataset_dir, self.example_names[idx])
        )

        return self.transform(input_tensor), self.transform(target_tensor)


class SegmentationDataset(pl.LightningDataModule):
    def __init__(self, *, batch_size):
        super().__init__()
        self.n_workers = min(os.cpu_count(), MAX_CPUS)
        self.dataset = FileDataset(
            SEGMENTATION_DATASET_DIR, os.listdir(SEGMENTATION_DATASET_DIR)
        )

        self.batch_size = batch_size
        self.train_set = None
        self.val_set = None

    def setup(self, stage=None):
        """
        This method sets up the training, validating and test datasets. This method is
        called on every subprocess when initializing the DataLoaders.

        :param stage: stage during which this method was called
        """

        train_set_size = int(len(self.dataset) * 0.8)
        valid_set_size = len(self.dataset) - train_set_size
        self.train_set, self.val_set = random_split(
            self.dataset, [train_set_size, valid_set_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
        )


class SegmentationUNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, u_depth=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.u_depth = u_depth

        current_channels = 64
        self.inc = UNetDoubleConv(n_channels, current_channels)

        # Create the 'down-side layers' of the U
        for i in range(u_depth):
            next_channels = current_channels * 2
            setattr(self, f"down{i}", UNetDownPass(current_channels, next_channels))
            current_channels = next_channels

        # Create the 'up-side layers' of the U
        for i in range(u_depth):
            next_channels = current_channels // 2
            setattr(self, f"up{i}", UNetUpPass(current_channels, next_channels))
            current_channels = next_channels

        self.conv_out = nn.Conv2d(current_channels, n_classes, kernel_size=1)

        nn.utils.clip_grad_value_(self.parameters(), GRADIENT_CLIP_VALUE)

    def forward(self, x):
        down_outputs = []
        down_outputs.append(self.inc(x))

        for i in range(self.u_depth):
            down_layer = getattr(self, f"down{i}")
            down_outputs.append(down_layer(down_outputs[-1]))

        x = down_outputs[-1]
        for i in range(self.u_depth):
            up_layer = getattr(self, f"up{i}")
            output_idx = self.u_depth - i - 1
            x = up_layer(x, down_outputs[output_idx])

        return self.conv_out(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = F.cross_entropy(predictions, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = F.cross_entropy(predictions, targets)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        av = torch.stack(validation_step_outputs).mean().item()
        logging.info(f"Average validation loss at epoch {self.current_epoch}: {av}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]


class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetDownPass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2), UNetDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UNetUpPass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.double_conv = UNetDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


def main(args):
    dataset = SegmentationDataset(batch_size=args.batch_size)
    model = SegmentationUNet(1, 3)

    trainer = pl.Trainer(
        gpus=-1,
        auto_select_gpus=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        logger=False,
    )

    trainer.fit(model, dataset)
    torch.save(model.state_dict(), "segmentation_model.pt")


if __name__ == "__main__":
    logging.getLogger("lightning").setLevel(logging.INFO)
    logging.getLogger("skl2onnx").setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    main(args)
