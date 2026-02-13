# Instance-segmentation
Mask R-CNN
masked R-CNN Model 
Backbone (feature extractor) -> ResNet-101 or ResNet-50 along with feature pyramid network (FPN)
region Proposal network (RPN) - CNN layer than scans the feature map to propose candidate object bounding boxes
ROIAlign - Replaces ROIPooling to accurately align features without pixel quantization, crucial for precise masks.
Head ( Prediction layers)
1. Class/ Box Head - Predicts the object class and refines the bounding box
2. Mask Head - A fully cnn applied to each ROI to predict a pixel level binary mask

Build the Architecture: Define the ResNet+FPN backbone, RPN, ROIPooling/Align, and the three classification/box/mask heads using frameworks like PyTorch or TensorFlow/Keras.
Define Loss Function: The total loss is a combination of RPN loss, classification loss, bounding box regression loss, and mask loss (
).
Training: Train on the custom dataset using GPU acceleration (e.g., Google Colab) for about 30+ epochs, monitoring convergence.
Inference: Use the trained model to perform detection, drawing bounding boxes, labels, and masks on new image



//////////////////////////////////
Mask RCNN with Python

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

#root directory of the project
ROOT_DIR = os.path.abspath("./")

#import MASK RCNN
sys.path.append(ROOT_DIR) #to find local version of the library
from mrcnn import utils
import mrcnn.model as matplotlib
from mrcnn import visualize
#import COO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/")) #to find local version
import coco

%matplotlib inline

#Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#Download coc trained weights from releases if needed
if not os.path.exists(COCO_MODEL_PATH):
       utils.download_trained_weights(COCO_MODEL_PATH)
      
#Directory of images to run detection on
IMAGE_DIR= os.path.join(ROOT_DIR, "images")

///////////////////////

////////////////////////

