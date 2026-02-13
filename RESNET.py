Prebuilt Resnet layer

--------------------------------------------
1. The Region Proposal Network (RPN)
The RPN slides over the feature map produced by your backbone. For every point in the feature map, it predicts whether an "anchor" contains an object and how to shift that anchor to fit the object better.

The Logic
Input: Feature map from the backbone (e.g., shape [C,H,W]).

Shared Convolution: A 3×3 layer to "gather" local context.

Classification Head: A 1×1 convolution that outputs 2×A (where A is the number of anchors per pixel) to represent "foreground" vs "background."

Regression Head: A 1×1 convolution that outputs 4×A to represent the offsets (Δx,Δy,Δw,Δh).

```
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
```       
