# Image Classification
Transfer learning-based image classification pipeline which generates heatmaps for interpretability and clinical analysis. We used this pipeline for training models for detection of any and referable pterygiums from colour anterior segment photographs. We also validated our models on an external dataset with handheld camera photographs.

# Basic Usage
The pipeline takes data as a pandas DataFrame with image file names and corresponding labels. Check ImageDataset for input details. Check sample_configs folder for detailed config files.

## Validation
```bash
python3 test.py  path/to/state_dicts/state_dict.pth
                 -c MAIN_DIR/config_test.json
                 -s path/to/store/validation_images
```
## Training
```bash
python3 train.py -c MAIN_DIR/config.json
```
## View logs
```bash
tensorboard --logdir MAIN_DIR/saved/logs/
```

# Training modifications
## Class imbalance
Add class_weights flag to change penalties if there are imbalanced training samples. Can specify an array with some classes or "weighted" to calculate automatically based on inverse number of samples.


## Image augmentation
Rotate, crop, move, shear or zoom in on image based on the torchvision [RandomAffine transform](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomAffine)

## Visualization

Visualization implements GradCAM & GradCAM++

## Early stopping

Early stopping based on validation loss

# Code references

1. Training pipeline based on [pytorch training template](https://github.com/victoresque/pytorch-template)
2. GradCAM, Gradcam++ visualization based on the [gradcam_plus_plus-pytorch](https://github.com/1Konny/gradcam_plus_plus-pytorch) repo
