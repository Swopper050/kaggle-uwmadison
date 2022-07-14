import glob

import matplotlib.pyplot as plt
import numpy as np
from uwm.constants import DATA_DIR


def id_from_path(path):
    """
    Given a path to an image, extract the id from as we would encounter it in the
    training dataframe. For example if the path is:
    "../input/uw-madison-gi-tract-image-segmentation/train/case101/case101_day20/scans/slice_0001_266_266_1.50_1.50.png"
    The ID will be: "case101_day20_slice_0001"

    :param path: str, path to an image
    :returns: id
    """
    path_parts = path.split("/")
    image_name = path_parts[-1].split(".")[0]
    slice_part = "_".join(image_name.split("_")[:2])
    return f"{path_parts[-3]}_{slice_part}"


def get_image_from_id(id):
    """
    Given an id string as found in the train dataframe, returns the image corresponding
    to that id.

    :param id: str, id of the train example
    :returns: np.array with the image
    :raises ValueError: if multiple files are found that match the id
    """

    case, day, _, slice_nr = id.split("_")
    img_path = f"{DATA_DIR}/train/{case}/{case}_{day}/scans/slice_{slice_nr}_*.png"
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


def rle_encode_mask(mask):
    """
    Given a mask, encode it in Run-Length Encoding (RLE). In RLE format, the mask will
    be a string that contains pairs of numbers. Every pair denotes the start index and
    the number of pixels from that index that are part of the mask. For example, suppose
    we have a mask like
    [[0.0, 1.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 1.0],]
    The RLE encoded mask will be "1 3 7 2".

    :param mask: 2D np.ndarray with 0 and 1 values
    :returns: str with RLE encoded mask
    """

    pixels = mask.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)
