import os

import numpy as np
import torch
from uwm.constants import PRESENT_DATASET_DIR, SEGMENTATION_DATASET_DIR
from uwm.utils import get_image_from_id, get_mask_from_rle


def write_input_and_targets_from_sample(id, sample, i):
    """
    Given a sample, which is a DataFrame row of our pivoted train csv, read in the
    image, get the masks (which will be our target) and whether or not the class is
    present from the segmentation information and write the tensor pairs to disk.
    Creates both a dataset for predicting whether an intestine is present or not and a
    dataset for image segmentation.

    :param id: str, id of the sample
    :param sample: pd.DataFrame with columns for all three segmentation classes
    :param i: int, the integer index of the row, used when saving the file
    """
    img = get_image_from_id(sample.name)

    large_bowel_mask = get_mask_from_rle(sample["large_bowel"], img.shape)
    small_bowel_mask = get_mask_from_rle(sample["small_bowel"], img.shape)
    stomach_mask = get_mask_from_rle(sample["stomach"], img.shape)
    background_mask = (large_bowel_mask + small_bowel_mask + stomach_mask) == 0
    segmentation_target = np.stack(
        [large_bowel_mask, small_bowel_mask, stomach_mask, background_mask], axis=0
    )
    present_target = segmentation_target.max(axis=1).max(axis=1)[:3]

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
