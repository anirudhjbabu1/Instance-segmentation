import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 1. Define the Backbone (Pre-built)
backbone = torchvision.models.resnet50(weights="DEFAULT")
# Remove the classifier head, keep the features
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
backbone.out_channels = 2048 # ResNet50 output channels

# 2. Define the Anchor Generator
# This tells the RPN which box sizes to use for each pixel in the grid
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# 3. Assemble the Mask R-CNN
# We use the official base class but can override components
model = MaskRCNN(
    backbone,
    num_classes=2, # Background + Your Object
    rpn_anchor_generator=anchor_generator
)

# You can manually assign your custom RPN or Mask Head here:
# model.rpn = CustomRPN(...)
# model.roi_heads.mask_head = CustomMaskHead(...)
-------------------------------------------------------------------------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # The model returns a dictionary of losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
       
        print(f"Loss: {losses.item()}")

# Usage: train_one_epoch(model, optimizer, train_data_loader, device)
--------------------------------------------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    # Pass a test image
    prediction = model([test_image.to(device)])

# Results are in prediction[0]
# prediction[0]['masks'] contains the soft-masks (0.0 to 1.0)
# prediction[0]['boxes'] contains [x1, y1, x2, y2]
# prediction[0]['scores'] contains confidence levels
----------------------------------------------

