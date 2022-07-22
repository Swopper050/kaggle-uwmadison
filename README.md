# UW-Madison GI Tract Image Segmentation

This repository contains all the code I used for competing in the
[UW-Madison Gi Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation).

#### TLDR; given an image, which is part of a medical scan, segment the intestines. The goal was to output image masks for all three classes: large bowel, small bowel and stomach.


![image](https://user-images.githubusercontent.com/38559175/180482469-f6ee8432-f6c0-4588-a794-0255c9f9cc97.png)


This competitation was about image segmentation. The goal was to, given an input image, segment the intestines in the image, and in particular the following three classes:
 - large bowel
 - small bowel
 - stomach
 
In order to generate the masks, I used a two stage process.

 1. First, I trained a model which predicts for every class whether or not it is present in the current image as it can be the case that none of the classes are present, or only 1 of them. For this I used a simple Convolutional Neural Network, the `resnet18` model from `torchvision`, finetuned on the dataset. 
 2. Then, I used the UNet architecture to generate 4 output masks, where 1 mask is the 'background' mask and the other three masks correspond to the classes to predict. This model was trained from scratch.
 3. The final prediction is either an empty mask if the first model predicts that the class is not present, or otherwise the output of the segmentation network.


# Installation
To reproduce the results, first clone the repository and install the dependencies (I used Python 3.10):

```bash
python -m venv .env
source .env/bin/activate
make deps
```

# Download data
Then download and extract the data. Make sure you have [setup the kaggle command line interface](https://www.kaggle.com/docs/api) and that you enable the download script `chmod +x download_data.sh`:
```bash
./download_data.sh
```

# Train models and generate submission
Two models need to be trained. The model that predicts whether the classes are present or not can be trained with:
```bash
python train_present_model.py --batch-size 32 --epochs 30
```

The model that performs the image segmentation can be trained with:
```bash
python train_segment_model.py --batch-size 16 --epochs 100
```

Both models are saved in the `models/` directory. Once both models are trained you can visually inspect its performance using:
```bash
python inspect_models.py
```

This script takes random training examples and plots the predictions for each class.

Finally, the script used to generate a submission can be runned with:
```bash
python create_submission.py
```
