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
