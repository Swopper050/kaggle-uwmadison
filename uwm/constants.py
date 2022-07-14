import torch

DATA_DIR = "./input/uw-madison-gi-tract-image-segmentation"
""" Directory where the dataset for image segmentation is stored. """
""" Base directory where the data is. """

PRESENT_PREDICTION_THRESHOLD = 0.5
""" Threshold for classifying whether or not an intestine is present. """

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASSES = ["large_bowel", "small_bowel", "stomach"]
