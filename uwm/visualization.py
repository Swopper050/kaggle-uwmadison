import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from uwm.utils import get_image_from_id, get_mask_from_rle


def custom_cmap(rgb_color):
    """A custom colormap used for visualization."""
    red, green, blue = rgb_color

    cdict = {
        "red": ((0, red, red), (1, red, red)),
        "green": ((0, green, green), (1, green, green)),
        "blue": ((0, blue, blue), (1, blue, blue)),
    }

    cmap = LinearSegmentedColormap("custom_cmap", cdict)
    return cmap


def plot_sample(sample):
    """
    Given a sample representing a row out of a train/test dataframe, plot the sample's
    image and its masks (the ground truth).

    :param sample: row of the train pd.DataFrame
    """
    img = get_image_from_id(sample.name)

    large_bowel_mask = get_mask_from_rle(sample["large_bowel"], img.shape)
    small_bowel_mask = get_mask_from_rle(sample["small_bowel"], img.shape)
    stomach_mask = get_mask_from_rle(sample["stomach"], img.shape)

    large_bowel_mask = np.ma.masked_where(large_bowel_mask == 0, large_bowel_mask)
    small_bowel_mask = np.ma.masked_where(small_bowel_mask == 0, small_bowel_mask)
    stomach_mask = np.ma.masked_where(stomach_mask == 0, stomach_mask)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # Original image
    ax1.set_title("Original Image")
    ax1.imshow(img)
    ax1.axis("off")

    ax2.set_title("Image with Mask")
    plot_masks(ax2, img, large_bowel_mask, small_bowel_mask, stomach_mask)

    plt.show()


def plot_prediction(img, target, pred, alpha=0.9):
    """
    Plots the predictions of the model versus the targets. It plots three images, one
    if the original image, one if the target masks and one with the predicted masks.

    :param img: 2d np.ndarray with the original image
    :param target: 3d np.ndarray with three channels representing the three target masks
    :param pred: 3d np.ndarray with three channels representing the three predicted masks
    """
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 10))

    ax1.set_title("Original Image")
    ax1.imshow(img)
    ax1.axis("off")

    target = np.ma.masked_where(target == 0, target)
    ax2.set_title("Image with true Mask")
    plot_masks(ax2, img, target[0], target[1], target[2])

    pred = np.ma.masked_where(pred < 0.5, pred)
    ax3.set_title("Image with predicted Mask")
    plot_masks(ax3, img, pred[0], pred[1], pred[2])

    ax3.imshow(pred[3], interpolation="none", alpha=0.4)

    plt.show()


def plot_masks(ax, img, large_bowel_mask, small_bowel_mask, stomach_mask, alpha=1.0):
    """
    Given a matplotlib axis, plot the image with the given masks on it. The original
    image is plotted as 'background' and the masks are plotted with different colors.

    :param ax: matplotlib axis
    :param img: 2d np.ndarray with the original image
    :param large_bowel_mask: 2d np.ndarray with the large bowel mask
    :param small_bowel_mask: 2d np.ndarray with the small bowel mask
    :param stomach_mask: 2d np.ndarray with the stomach mask
    :param alpha: float, how transparent the masks are plotted
    """
    mask_colors = [(1.0, 0.7, 0.1), (1.0, 0.5, 1.0), (1.0, 0.22, 0.099)]
    legend_colors = [Rectangle((0, 0), 1, 1, color=color) for color in mask_colors]
    labels = ["Large Bowel", "Small Bowel", "Stomach"]

    cmap1 = custom_cmap(mask_colors[0])
    cmap2 = custom_cmap(mask_colors[1])
    cmap3 = custom_cmap(mask_colors[2])

    ax.imshow(img)
    ax.imshow(large_bowel_mask, interpolation="none", cmap=cmap1, alpha=alpha)
    ax.imshow(small_bowel_mask, interpolation="none", cmap=cmap2, alpha=alpha)
    ax.imshow(stomach_mask, interpolation="none", cmap=cmap3, alpha=alpha)
    ax.legend(legend_colors, labels)
    ax.axis("off")
