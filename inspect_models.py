import argparse

import torch
from uwm.constants import CLASSES
from uwm.dataset import SegmentationDatamodule
from uwm.present_model import PresentPredictor
from uwm.segmentation_model import SegmentationUNet
from uwm.visualization import plot_prediction


def main(args):
    dataset = SegmentationDatamodule(batch_size=1)

    present = PresentPredictor()
    present.load_state_dict(torch.load("./models/present_model.pt"))
    present.eval()

    segment = SegmentationUNet(n_channels=1, n_classes=4, u_depth=4, start_channels=32)
    segment.load_state_dict(torch.load("./models/segmentation_model.pt"))
    segment.eval()

    dataset.setup()
    for batch in dataset.train_dataloader():
        inputs, targets = batch
        with torch.no_grad():
            present_preds = present(inputs.repeat(1, 3, 1, 1))[0].numpy()
            masks = segment.predict_masks(inputs)[0]

        present_preds = {
            class_name: round(present_preds[i], 2)
            for i, class_name in enumerate(CLASSES)
        }

        plot_prediction(inputs[0][0].numpy(), targets[0].numpy(), masks, present_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
