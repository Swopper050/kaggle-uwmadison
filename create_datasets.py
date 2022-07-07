import argparse
import os
import shutil

import pandas as pd
from uwm.constants import BASE_DIR, PRESENT_DATASET_DIR, SEGMENTATION_DATASET_DIR
from uwm.data import write_input_and_targets_from_sample


def main(args):
    if os.path.exists(SEGMENTATION_DATASET_DIR) or os.path.exists(PRESENT_DATASET_DIR):
        choice = input("Existing data, do you want to remove it? [y/N] ")
        if choice.lower() not in ["y", "yes"]:
            print("exiting")
            exit()

        shutil.rmtree(SEGMENTATION_DATASET_DIR)
        shutil.rmtree(PRESENT_DATASET_DIR)

    os.makedirs(SEGMENTATION_DATASET_DIR)
    os.makedirs(PRESENT_DATASET_DIR)

    train = pd.read_csv(f"{BASE_DIR}/train.csv")
    train = train.pivot(index="id", columns="class", values="segmentation")

    n_samples = len(train)
    for i, (id, sample) in enumerate(train.iterrows()):
        write_input_and_targets_from_sample(id, sample, i)
        print(f"Wrote {i}/{n_samples} samples", end="\r")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
