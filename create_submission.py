import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from uwm.constants import CLASSES, DATA_DIR, DEVICE, PRESENT_PREDICTION_THRESHOLD
from uwm.present_model import PresentPredictor
from uwm.segmentation_model import SegmentationUNet
from uwm.utils import id_from_path, rle_encode_mask

USE_PRESENT_PREDICTOR = True


def get_center_crop_slices(img_size, crop_size=(266, 266)):
    """
    Given an image and a desired size, creates slices that can be used to access the
    centered crop for this image given this size.

    :param img: tuple, image size
    :param size: tuple, desired size of the cropped image
    :returns: 2 slices that can be used to select a center crop from an image
    """

    height, width = img_size[0], img_size[1]
    height_diff = height - crop_size[0]
    width_diff = width - crop_size[1]

    top_offset = height_diff // 2
    down_offset = height - (crop_size[0] + top_offset)

    left_offset = width_diff // 2
    right_offset = width - (crop_size[1] + left_offset)

    return (
        slice(top_offset, (height - down_offset)),
        slice(left_offset, (width - right_offset)),
    )


def get_image_paths():
    """
    Returns a list with paths to images that will be used to create a submission.
    The test set is only available during submission, hence for debugging we will
    select a few train images to create a submission for.

    :returns: list of str, paths to images
    """
    path = os.path.join(DATA_DIR, "test", "**", "*.png")
    image_paths = glob.glob(path, recursive=True)

    if len(image_paths) == 0:
        image_paths = glob.glob(
            os.path.join(DATA_DIR, "train", "**", "*.png"), recursive=True
        )[:1000]

    return image_paths


def main(args):
    present = PresentPredictor()
    present.load_state_dict(torch.load("./models/present_model.pt"))
    present.to(DEVICE)
    present.eval()

    segment = SegmentationUNet(n_channels=1, n_classes=4, u_depth=4, start_channels=32)
    segment.load_state_dict(torch.load("./models/segmentation_model.pt"))
    segment.to(DEVICE)
    segment.eval()

    results = []
    image_paths = get_image_paths()
    for img_path in image_paths:
        id = id_from_path(img_path)
        img = plt.imread(img_path)
        height_slice, width_slice = get_center_crop_slices(
            img.shape, crop_size=(266, 266)
        )

        cropped_img = (
            torch.from_numpy(img[height_slice, width_slice])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            present_preds = present(cropped_img.repeat(1, 3, 1, 1))[0]
            predicted_masks = segment.predict_masks(cropped_img)[0]

        for i, class_name in enumerate(CLASSES):
            if USE_PRESENT_PREDICTOR and present_preds[i] < PRESENT_PREDICTION_THRESHOLD:
                rle_mask = ""
            else:
                mask = predicted_masks[i]
                full_size_mask = np.zeros(img.shape)
                full_size_mask[height_slice, width_slice] = mask
                rle_mask = rle_encode_mask(full_size_mask)

            results.append({"id": id, "class": class_name, "predicted": rle_mask})

    df = pd.DataFrame.from_records(results)
    print(df.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
