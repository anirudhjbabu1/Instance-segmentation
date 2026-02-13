-----------------------------------------------------
2. The Mask HeadAfter RoI Align crops and resizes a proposal to a fixed size (usually $14 \times 14$), it passes through the Mask Head. Unlike the classification head, this must stay convolutional to preserve spatial information.The LogicStack of Convolutions: Usually 4 layers of $3 \times 3$ convolutions with ReLU. This allows the network to "see" the shape of the object within the crop.Deconvolution (ConvTranspose): This upsamples the $14 \times 14$ feature map to $28 \times 28$.Final Predictor: A $1 \times 1$ convolution with a Sigmoid activation. This gives you a pixel-by-pixel probability map.

```
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
```       
