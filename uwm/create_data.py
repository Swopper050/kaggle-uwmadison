import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
from uwm.constants import BASE_DIR, PRESENT_DATASET_DIR, SEGMENTATION_DATASET_DIR
from uwm.utils import get_image_from_id, get_mask_from_rle


def write_input_and_target_from_sample(id, sample, i):
    """
    Given a sample, which is a DataFrame row of our pivoted train csv, read in the
    image, get the masks (which will be our target) from the segmentation information
    and write the tensor pair to disk, which will be used later as the dataset.

    :param id: str, id of the sample
    :param sample: pd.DataFrame with columns for all three segmentation classes
    :param i: int, the integer index of the row, used when saving the file
    """
    img = get_image_from_id(sample.name)

    large_bowel_mask = get_mask_from_rle(sample["large_bowel"], img.shape)
    small_bowel_mask = get_mask_from_rle(sample["small_bowel"], img.shape)
    stomach_mask = get_mask_from_rle(sample["stomach"], img.shape)
    segmentation_target = np.stack(
        [large_bowel_mask, small_bowel_mask, stomach_mask], axis=0
    )
    present_target = segmentation_target.max(axis=1).max(axis=1)

    input_tensor = torch.from_numpy(img.reshape([1, *img.shape])).float()
    segmentation_target_tensor = torch.from_numpy(segmentation_target).float()
    present_target_tensor = torch.from_numpy(present_target).float()

    torch.save(
        (input_tensor, segmentation_target_tensor),
        os.path.join(SEGMENTATION_DATASET_DIR, f"{i}.pt"),
    )
    torch.save(
        (input_tensor, present_target_tensor),
        os.path.join(PRESENT_DATASET_DIR, f"{i}.pt"),
    )


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
        write_input_and_target_from_sample(id, sample, i)
        print(f"Wrote {i}/{n_samples} samples", end="\r")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
