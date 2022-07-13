import argparse
import logging
import sys

import pytorch_lightning as pl
import torch
from uwm.dataset import PresentDatamodule
from uwm.present_model import PresentPredictor


def main(args):
    dataset = PresentDatamodule(batch_size=args.batch_size)
    model = PresentPredictor()

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        logger=False,
    )

    trainer.fit(model, dataset)
    torch.save(model.state_dict(), "./present_model.pt")


if __name__ == "__main__":
    logging.getLogger("lightning").setLevel(logging.INFO)
    logging.getLogger("skl2onnx").setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
