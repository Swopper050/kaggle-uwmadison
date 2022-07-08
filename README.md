# UW-Madison GI Tract Image Segmentation

This repository contains all the code I used for competing in the
[UW-Madison Gi Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation).

# Installation
To reproduce the results, first clone the repository and install the dependencies (I used Python 3.10):

```bash
python -m venv .env
source .env/bin/activate
python -m pip install requirements.txt
```

Then download and extract the data (first [setup the kaggle command line interface](https://www.kaggle.com/docs/api)):
```bash
kaggle competitions download -c uw-madison-gi-tract-image-segmentation
mv uw-madison-gi-tract-image-segmentation.zip input/
cd input
mkdir uw-madison-gi-tract-image-segmentation
unzip -d uw-madison-gi-tract-image-segmentation uw-madison-gi-tract-image-segmentation.zip
rm uw-madison-gi-tract-image-segmentation.zip
```

# Create datasets
To create the datasets for training the neural networks run:
```bash
python create_datasets.py
```
