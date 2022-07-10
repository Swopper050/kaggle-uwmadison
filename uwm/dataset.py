import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from uwm.constants import DATA_DIR
from uwm.utils import get_image_from_id, get_mask_from_rle

MAX_CPUS = 8
"""Maximum number of CPU cores to use for loading data examples. """


class FileDataset(Dataset):
    """
    Extended from the torch.utils.data.Dataset, this class can be used in combination
    with a Dataloader to lazily load the dataset using files as examples.
    """

    def __init__(self):
        """
        :param dataset_dir: str, path to the folder with all examples
        :param example_names: list of str with all example names
        """
        self.train = pd.read_csv(f"{DATA_DIR}/train.csv").pivot(
            index="id", columns="class", values="segmentation"
        )
        self.transform = transforms.Compose(
            [
                # Shape (266, 266) because it is most representative
                transforms.RandomCrop((266, 266), pad_if_needed=True),
                transforms.RandomRotation(45),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.3),
            ]
        )

    def __len__(self):
        """
        :returns: the total number of examples in the dataset
        """
        return len(self.train)

    def __getitem__(self, idx):
        """
        Given an index, retrieves the sample from the training csv file. Given this
        sample, which is a DataFrame row of our pivoted train csv, read in the
        image, get the masks (which will be our target), transform the input and target
        masks and return them as torch tensors.

        :param idx: int, specifying which record to load
        :returns: encoder input, decoder input, targets as tensors
        """
        sample = self.train.iloc[idx]
        img = get_image_from_id(sample.name)
        input_tensor = torch.from_numpy(img.reshape([1, *img.shape])).float()

        large_bowel_mask = get_mask_from_rle(sample["large_bowel"], img.shape)
        small_bowel_mask = get_mask_from_rle(sample["small_bowel"], img.shape)
        stomach_mask = get_mask_from_rle(sample["stomach"], img.shape)
        masks = np.stack([large_bowel_mask, small_bowel_mask, stomach_mask], axis=0)

        masks_tensor = self.transform(torch.from_numpy(masks).float())
        background_mask = masks_tensor.sum(axis=0) == 0
        target_tensor = torch.cat((masks_tensor, background_mask.unsqueeze(0)), dim=0)

        return self.transform(input_tensor), target_tensor


class SegmentationDatamodule(pl.LightningDataModule):
    """
    Uses the FileDataset to create a LightningDataModule that is used to train the
    segmentation network. It is a wrapper around the dataset that handles the train
    and validation datasets.
    """

    def __init__(self, *, batch_size):
        super().__init__()
        self.n_workers = min(os.cpu_count(), MAX_CPUS)
        self.dataset = FileDataset()

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
