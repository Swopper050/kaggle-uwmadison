import torch

DATA_DIR = "./input/uw-madison-gi-tract-image-segmentation"
""" Directory where the dataset for image segmentation is stored. """
""" Base directory where the data is. """

PREDICTION_THRESHOLD = 0.5
""" Threshold for pixel probabilities to be classified as the class. """

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
