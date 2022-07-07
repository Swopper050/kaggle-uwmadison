import argparse
import logging
import sys

import pytorch_lightning as pl
import torch
from uwm.present_model import PresentDataset, PresentPredictor


def main(args):
    dataset = PresentDataset(batch_size=args.batch_size)
    model = PresentPredictor()

    trainer = pl.Trainer(
        gpus=-1,
        auto_select_gpus=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        logger=False,
    )

    trainer.fit(model, dataset)
    torch.save(model.state_dict(), "present_model.pt")


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
