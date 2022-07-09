import argparse

import torch
import torch.nn.functional as F
from uwm.segmentation_model import SegmentationDataset, SegmentationUNet
from uwm.visualization import plot_prediction


def main(args):
    dataset = SegmentationDataset(batch_size=1)
    model = SegmentationUNet(n_channels=1, n_classes=4, u_depth=2)
    model.load_state_dict(torch.load("./models/segmentation_model.pt"))

    dataset.setup()
    for batch in dataset.train_dataloader():
        inputs, targets = batch
        with torch.no_grad():
            preds = model(inputs)

        loss = F.cross_entropy(preds, targets)
        print(loss)
        plot_prediction(inputs[0][0].numpy(), targets[0].numpy(), preds[0].numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
