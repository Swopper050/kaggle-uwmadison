import argparse
import logging
import sys

import pytorch_lightning as pl
import torch
from uwm.dataset import SegmentationDatamodule
from uwm.segmentation_model import SegmentationUNet


def main(args):
    dataset = SegmentationDatamodule(batch_size=args.batch_size)
    model = SegmentationUNet(n_channels=1, n_classes=4, u_depth=4, start_channels=32)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        logger=False,
    )

    trainer.fit(model, dataset)
    torch.save(model.state_dict(), "./models/segmentation_model.pt")


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
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
