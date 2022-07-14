import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class PresentPredictor(pl.LightningModule):
    """
    The PresentPredictor predicts for each class (large_bowel, small_bowel, stomach)
    whether or not it is present in the given image. It uses a pretrained resnet
    model, which is finetuned during training.
    """

    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        n_inputs_last_layer = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(n_inputs_last_layer, 3),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = F.binary_cross_entropy(predictions, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        soft_predictions = self(inputs)
        loss = F.binary_cross_entropy(soft_predictions, targets)

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
