import torch

DATA_DIR = "./input/uw-madison-gi-tract-image-segmentation"
""" Directory where the dataset for image segmentation is stored. """
""" Base directory where the data is. """

SEGMENTATION_DATASET_DIR = "./segmentation_dataset"
""" Directory where the dataset for image segmentation is stored. """

PRESENT_DATASET_DIR = "./present_dataset"
""" Directory where the dataset for 'present intestines' is stored. """

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
