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

Prebuilt Resnet layer

--------------------------------------------
1. The Region Proposal Network (RPN)
The RPN slides over the feature map produced by your backbone. For every point in the feature map, it predicts whether an "anchor" contains an object and how to shift that anchor to fit the object better.

The Logic
Input: Feature map from the backbone (e.g., shape [C,H,W]).

Shared Convolution: A 3×3 layer to "gather" local context.

Classification Head: A 1×1 convolution that outputs 2×A (where A is the number of anchors per pixel) to represent "foreground" vs "background."

Regression Head: A 1×1 convolution that outputs 4×A to represent the offsets (Δx,Δy,Δw,Δh).

import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels=256, anchors_per_location=9):
        super(RPN, self).__init__()
       
        # 3x3 Shared Convolutional layer
        self.conv_shared = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
       
        # Classification layer: 2 scores per anchor (Object vs Background)
        # Output shape: [Batch, anchors_per_location * 2, H, W]
        self.cls_logits = nn.Conv2d(in_channels, anchors_per_location * 2, kernel_size=1)
       
        # Regression layer: 4 coordinates per anchor (dx, dy, dw, dh)
        # Output shape: [Batch, anchors_per_location * 4, H, W]
        self.bbox_pred = nn.Conv2d(in_channels, anchors_per_location * 4, kernel_size=1)

    def forward(self, feature_map):
        x = F.relu(self.conv_shared(feature_map))
       
        logits = self.cls_logits(x)
        bbox_deltas = self.bbox_pred(x)
       
        return logits, bbox_deltas
       
-----------------------------------------------------
2. The Mask HeadAfter RoI Align crops and resizes a proposal to a fixed size (usually $14 \times 14$), it passes through the Mask Head. Unlike the classification head, this must stay convolutional to preserve spatial information.The LogicStack of Convolutions: Usually 4 layers of $3 \times 3$ convolutions with ReLU. This allows the network to "see" the shape of the object within the crop.Deconvolution (ConvTranspose): This upsamples the $14 \times 14$ feature map to $28 \times 28$.Final Predictor: A $1 \times 1$ convolution with a Sigmoid activation. This gives you a pixel-by-pixel probability map.

class MaskHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=81):
        super(MaskHead, self).__init__()
       
        # Stack of 4 convolutional layers to extract spatial features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU()
        )
       
        # Deconvolution (Upsampling) to go from 14x14 -> 28x28
        self.upsample = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
       
        # Final prediction layer: 1 mask per class
        self.mask_predictor = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, roi_features):
        # roi_features is the output of RoI Align (Size: [N, 256, 14, 14])
        x = self.conv_layers(roi_features)
        x = F.relu(self.upsample(x))
       
        # Probability map for each class (Size: [N, num_classes, 28, 28])
        masks = torch.sigmoid(self.mask_predictor(x))
        return masks
       
--------------------------------------------------
3. Implementing the Loss FunctionsSince you are writing the logic manually, you'll need to define how the model learns. The total loss is a weighted sum of several parts.RPN Loss$$L_{RPN} = L_{rpn\_cls} + L_{rpn\_bbox}$$$L_{rpn\_cls}$: Binary Cross Entropy (Is there an object here?).$L_{rpn\_bbox}$: Smooth L1 Loss (Only calculated for "positive" anchors that actually overlap with a real object).Mask Loss$$L_{mask} = \text{Binary Cross Entropy}$$Crucial Detail: The mask loss is only defined for the "true" class of the object. If the classifier says the object is a "Cat," you only calculate the loss on the "Cat" mask buffer, ignoring the others. This prevents different classes from competing for pixels.

def compute_mask_loss(pred_masks, target_masks, target_class_ids):
    """
    pred_masks: [N, num_classes, 28, 28]
    target_masks: [N, 28, 28] (Binary 0 or 1)
    target_class_ids: [N] (The actual class for each RoI)
    """
    # 1. Pick the mask corresponding to the correct class for each RoI
    # This is the "Mask Branch" trick in the original paper
    row_indices = torch.arange(len(target_class_ids))
    relevant_masks = pred_masks[row_indices, target_class_ids] # [N, 28, 28]
   
    # 2. Binary Cross Entropy Loss
    loss = F.binary_cross_entropy(relevant_masks, target_masks)
    return loss
----------------------------------------
4.Here is a simplified Python function to generate an anchor grid.


import numpy as np

def generate_anchors(feature_map_size, stride=16, scales=[32, 64, 128], ratios=[0.5, 1, 2]):
    """
    Generates a grid of anchor boxes.
    feature_map_size: (height, width) of the backbone output (e.g., 64, 64)
    stride: how many pixels in the input image correspond to 1 pixel in the feature map
    """
    anchors = []
   
    # 1. Loop through every pixel in the feature map
    for y in range(feature_map_size[0]):
        for x in range(feature_map_size[1]):
           
            # Calculate the center of the anchor in the original image
            center_x = (x + 0.5) * stride
            center_y = (y + 0.5) * stride
           
            # 2. For each location, create anchors of all scales and ratios
            for scale in scales:
                for ratio in ratios:
                    # Calculate width and height based on aspect ratio
                    h = scale * np.sqrt(ratio)
                    w = scale / np.sqrt(ratio)
                   
                    # Store as [y1, x1, y2, x2]
                    anchors.append([
                        center_y - h/2, center_x - w/2,
                        center_y + h/2, center_x + w/2
                    ])
                   
    return np.array(anchors)

# Example usage for a 1024x1024 image with a ResNet backbone (stride 16)
anchors = generate_anchors((64, 64))
print(f"Total anchors generated: {len(anchors)}")
