import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

GRADIENT_CLIP_VALUE = 0.1
"""Clip gradients during training to this number to avoid large gradients. """

PREDICTION_THRESHOLD = 0.5
""" Probability predictions larger than this value will be predicted as true. """


def get_dice_coefficient(preds, targets):
    """
    Calculates the discrete dice coefficient given predictions and targets. Works for
    batches as well. Assumes a shape of [B, 1, H, W]. It is important that only one
    channels is passed (the 1 in the shape), as the dice coefficient assumes binary
    classes. This method should be used to calculate class wice dice coefficients,
    which can be averaged afterwards.
    If the target contains no positive labels and the predictions also don't contain
    positive labels, a dice coefficient of 1 is returned.

    :param preds: torch.tensor of shape [B, 1, H, W]
    :param targets: torch.tensor of shape [B, 1, H, W]
    :returns: torch.tensor with the dice coefficient
    """
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    total_preds = preds.sum()
    total_targets = targets.sum()
    if total_preds == 0 and total_targets == 0:
        return 1.0

    return (2 * (preds * targets).sum()) / (total_preds + total_targets)


def get_continuous_dice_coefficient(preds, targets):
    """
    Calculates the continuous version of the dice coefficient. Basically same as the
    discrete dice coefficient, but in the numerator we multiply by a coefficient to
    ensure we keep more information about the magnitude of the probabilities, which
    is useful when using the dice coefficient as training loss.

    :param preds: torch.tensor of shape [B, 1, H, W]
    :param targets: torch.tensor of shape [B, 1, H, W]
    :returns: torch.tensor with the dice coefficient
    """
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    total_preds = preds.sum()
    total_targets = targets.sum()
    if total_preds == 0 and total_targets == 0:
        return 1.0

    intersection = (preds * targets).sum()

    x = (preds * torch.sign(targets)).sum()
    c = intersection / x if x > 0 else 1
    return (2 * intersection) / (c * total_preds + total_targets)


def get_multiclass_dice_coefficient(preds, targets, continuous=False):
    """
    Calculates the multiclass dice coefficient. Assumes `preds` and `targets`
    are of shape [B, C, H, W], where B is batch size and C is the number of
    classes. Calculates the dice coefficient for each class separately and
    averages the result. Depending on the use case (loss vs metric), the continuous
    dice coefficient can be used.

    :param preds: torch.tensor of shape [B, C, H, W]
    :param targets: torch.tensor of shape [B, C, H, W]
    :param continuous: bool, if True, uses the continuous dice coefficient calculation
    :returns: torch.tensor with the mult-class dice coefficient
    """
    n_classes = preds.shape[1]
    dice_fn = get_continuous_dice_coefficient if continuous else get_dice_coefficient

    total_dice_loss = 0.0
    for i in range(n_classes):
        total_dice_loss += dice_fn(preds[:, i], targets[:, i])

    return total_dice_loss / n_classes


def dice_loss(preds, targets):
    """
    Returns the dice loss, i.e. the multi class dice coefficient framed such that it
    can be minimized, i.e. (1 - dice_coef). Uses the continuous dice coefficient to
    preserver more information during training.

    :param preds: torch.tensor of shape [B, C, H, W]
    :param targets: torch.tensor of shape [B, C, H, W]
    :returns: torch.tensor with the dice loss
    """
    return 1.0 - get_multiclass_dice_coefficient(preds, targets, continuous=True)


class SegmentationUNet(pl.LightningModule):
    """UNet implementation for image segmentation."""

    def __init__(self, n_channels, n_classes, u_depth=3, start_channels=16):
        """
        :param n_channels: int, number of channels input images have
        :param n_classes: int, number of output masks
        :param u_depth: int, depth of the 'U'-shape. Deeper means more parameters
        :param start_channels: int, number of output channels for the first convolution
        """

        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.u_depth = u_depth

        current_channels = start_channels
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

        # Set gradient clipping to avoid large gradients.
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

        return torch.sigmoid(self.conv_out(x))

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        # loss = dice_loss(predictions, targets)
        weights = torch.tensor([1.0, 1.0, 1.0, 0.1], device=self.device)
        loss = F.cross_entropy(predictions, targets, weights)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        # loss = dice_loss(predictions, targets).item()
        weights = torch.tensor([1.0, 1.0, 1.0, 0.1], device=self.device)
        loss = F.cross_entropy(predictions, targets, weights).item()

        predictions[predictions > 0.5] = 1.0
        predictions[predictions < 0.5] = 0.0
        dice_coef = get_multiclass_dice_coefficient(predictions, targets).item()
        return {
            "val_loss": loss,
            "dice_coef": dice_coef,
        }

    def validation_epoch_end(self, validation_step_outputs):
        av_loss = np.mean([step["val_loss"] for step in validation_step_outputs])
        av_dice = np.mean([step["dice_coef"] for step in validation_step_outputs])
        logging.info(f"Average val loss at epoch {self.current_epoch}: {av_loss}")
        logging.info(f"Average val dice at epoch {self.current_epoch}: {av_dice}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class UNetDoubleConv(nn.Module):
    """
    Helper block for UNet. This module performs double convolution. It simply
    takes a number of input feature mas, performs 2 times convolution together
    with batch normalization and a ReLU activation function and returns the
    resulting feature maps.
    """

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
    """
    Represents a downward pass in the UNet network. It consists of first a simple
    MaxPooling operation, afterwards we perform the `UNetDoubleConv` pass again.

    This effectively scales down the inputs again (due to the pooling), and then
    performs new convolutions on the downscaled feature maps.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2), UNetDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UNetUpPass(nn.Module):
    """
    Performs a upward pass in the UNet network. Given a number of input feature maps,
    it first samples them up using `ConvTranspose2d`. Then it performs a
    `UNetDoubleConv` pass again. However, the upward pass also takes as input the
    output of the layer at the same height of the 'U', as a result of the skip
    connections of UNet. These feature maps are concatenated with the upsampled
    feature maps before the double convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.double_conv = UNetDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        :param x1: feature maps from the previous layers
        :param x2: feature maps resulting from the skip connection
        """
        x1 = self.up(x1)

        # Pad the feature maps so they can be concatenated with the feature maps
        # from the skip connection.
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)
