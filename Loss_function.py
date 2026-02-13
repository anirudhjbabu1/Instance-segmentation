--------------------------------------------------
3. Implementing the Loss FunctionsSince you are writing the logic manually, you'll need to define how the model learns. The total loss is a weighted sum of several parts.RPN Loss$$L_{RPN} = L_{rpn\_cls} + L_{rpn\_bbox}$$$L_{rpn\_cls}$: Binary Cross Entropy (Is there an object here?).$L_{rpn\_bbox}$: Smooth L1 Loss (Only calculated for "positive" anchors that actually overlap with a real object).Mask Loss$$L_{mask} = \text{Binary Cross Entropy}$$Crucial Detail: The mask loss is only defined for the "true" class of the object. If the classifier says the object is a "Cat," you only calculate the loss on the "Cat" mask buffer, ignoring the others. This prevents different classes from competing for pixels.

```
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
```
