import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from constants import PRESENT_DATASET_DIR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import CenterCrop

np.set_printoptions(precision=3, suppress=True)

MAX_CPUS = 8
"""Maximum number of CPU cores to use for loading dataset examples. """


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

        return self.transform(input_tensor).repeat(3, 1, 1), target_tensor


class PresentDataset(pl.LightningDataModule):
    def __init__(self, *, batch_size):
        super().__init__()
        self.n_workers = min(os.cpu_count(), MAX_CPUS)
        self.dataset = FileDataset(PRESENT_DATASET_DIR, os.listdir(PRESENT_DATASET_DIR))

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


class PresentPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        n_inputs_last_layer = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(n_inputs_last_layer, 3),
            torch.nn.Sigmoid(),
        )

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.model(inputs)
        loss = torch.nn.functional.binary_cross_entropy(predictions, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        soft_predictions = self.model(inputs)
        loss = torch.nn.functional.binary_cross_entropy(soft_predictions, targets)
        accuracy, precision, recall = class_wise_metrics(
            soft_predictions.round(), targets
        )
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

    def validation_epoch_end(self, validation_step_outputs):
        accuracies = (
            torch.stack([out["accuracy"] for out in validation_step_outputs])
            .mean(axis=0)
            .cpu()
            .numpy()
        )
        precisions = (
            torch.stack([out["precision"] for out in validation_step_outputs])
            .mean(axis=0)
            .cpu()
            .numpy()
        )
        recalls = (
            torch.stack([out["recall"] for out in validation_step_outputs])
            .mean(axis=0)
            .cpu()
            .numpy()
        )
        logging.info(f"Metrics at epoch {self.current_epoch}:")
        logging.info(f"Accuracies:\t{accuracies}")
        logging.info(f"Precisions:\t{precisions}")
        logging.info(f"Recalls:\t\t{recalls}\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        return [optimizer], [scheduler]


def class_wise_metrics(predictions, targets):
    """
    Calculates metrics (accuracy, precision and recall) for all classes individually.

    :param predictions: batch with predictions
    :param targets: ground truths
    :returns: 3 tensors, the metrics for each class
    """

    true_positives = ((predictions + targets) == 2).sum(axis=0)
    false_positives = ((predictions - targets) == 1).sum(axis=0)
    false_negatives = ((predictions - targets) == -1).sum(axis=0)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return class_wise_accuracy(predictions, targets), precision, recall


def class_wise_accuracy(predictions, targets):
    """Returns the class-wise accuracy for all three classes."""
    return (predictions == targets).float().mean(axis=0)
