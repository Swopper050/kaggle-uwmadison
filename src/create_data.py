import argparse
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from constants import BASE_DIR, PRESENT_DATASET_DIR, SEGMENTATION_DATASET_DIR


def CustomCmap(rgb_color):

    r1, g1, b1 = rgb_color

    cdict = {
        "red": ((0, r1, r1), (1, r1, r1)),
        "green": ((0, g1, g1), (1, g1, g1)),
        "blue": ((0, b1, b1), (1, b1, b1)),
    }

    cmap = LinearSegmentedColormap("custom_cmap", cdict)
    return cmap


def get_image_from_id(id):
    """
    Given an id string as found in the train dataframe, returns the image corresponding
    to that id.

    :param id: str, id of the train example
    :returns: np.array with the image
    :raises ValueError: if multiple files are found that match the id
    """

    case, day, _, slice_nr = id.split("_")
    img_path = f"{BASE_DIR}/train/{case}/{case}_{day}/scans/slice_{slice_nr}_*.png"
    paths = glob.glob(img_path)

    if len(paths) != 1:
        raise ValueError(f"Image path yielded {len(paths)} files: {img_path}")

    return plt.imread(paths[0])


def get_mask_from_rle(encoded_masks, img_shape):
    """
    Creates a binary mask with the given shape. `encoded_masks` is a space separated
    string of integers that, in pairs, denote mask pieces. The first number of a pair
    denotes the start index of a mask piece, the second number of a pair denotes the
    number of pixels after this start which belong to the mask piece as well.

    :param encoded_masks: list with integers, representing the run length encoded masks
    :param img_shape: shape of the image
    :returns: np.ndarray with the same shape as the image, but with the mask
    :raises ValueError: if `encoded_masks` has an uneven number of elements
    """

    mask = np.zeros(img_shape)

    if type(encoded_masks) != str:
        return mask

    pair_list = [int(val) for val in encoded_masks.split(" ")]
    if len(pair_list) % 2 != 0:
        raise ValueError("RLE mask contains uneven number of elements")

    mask = mask.flatten()
    for i in range(0, len(pair_list), 2):
        start = pair_list[i]
        length = pair_list[i + 1]
        mask[start : start + length] = 1.0

    return mask.reshape(img_shape)


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


def main1(args):
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


def plot_sample(sample, alpha=1):
    img = get_image_from_id(sample.name)

    large_bowel_mask = get_mask_from_rle(sample["large_bowel"], img.shape)
    large_bowel_mask = np.ma.masked_where(large_bowel_mask == 0, large_bowel_mask)

    small_bowel_mask = get_mask_from_rle(sample["small_bowel"], img.shape)
    small_bowel_mask = np.ma.masked_where(small_bowel_mask == 0, small_bowel_mask)

    stomach_mask = get_mask_from_rle(sample["stomach"], img.shape)
    stomach_mask = np.ma.masked_where(stomach_mask == 0, stomach_mask)

    mask_colors = [(1.0, 0.7, 0.1), (1.0, 0.5, 1.0), (1.0, 0.22, 0.099)]
    legend_colors = [Rectangle((0, 0), 1, 1, color=color) for color in mask_colors]
    labels = ["Large Bowel", "Small Bowel", "Stomach"]

    CMAP1 = CustomCmap(mask_colors[0])
    CMAP2 = CustomCmap(mask_colors[1])
    CMAP3 = CustomCmap(mask_colors[2])

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    # Original
    ax1.set_title("Original Image")
    ax1.imshow(img)
    ax1.axis("off")

    # With Mask
    ax2.set_title("Image with Mask")
    ax2.imshow(img)
    ax2.imshow(large_bowel_mask, interpolation="none", cmap=CMAP1, alpha=alpha)
    ax2.imshow(small_bowel_mask, interpolation="none", cmap=CMAP2, alpha=alpha)
    ax2.imshow(stomach_mask, interpolation="none", cmap=CMAP3, alpha=alpha)
    ax2.legend(legend_colors, labels)
    ax2.axis("off")

    plt.show()


def main(args):
    def rle_encode(img):
        """
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        """
        pixels = img.flatten()
        # pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return " ".join(str(x) for x in runs)

    train = pd.read_csv(f"{BASE_DIR}/train.csv")
    train = train.pivot(index="id", columns="class", values="segmentation")

    sample = train.iloc[515]
    img = get_image_from_id(sample.name)

    large_bowel_mask = get_mask_from_rle(sample["large_bowel"], img.shape)
    result = rle_encode(large_bowel_mask)

    print(img.shape)
    print("Q:", result)
    print("O:", sample["large_bowel"])

    plt.imshow(large_bowel_mask)
    plt.show()

    __import__("pdb").set_trace()

    # plot_sample(sample, alpha=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
