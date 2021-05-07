import numpy as np

def miou(preds, gt, num_classes):
    ious = []
    pred = preds.view(-1)
    target = gt.view(-1)

    # Ignore IoU for background class ("0")
    # This goes from 1:n_classes-1 -> class "0" is ignored
    for cls in range(1, num_classes):  
        pred_inds = pred == cls
        target_inds = target == cls
        # Cast to long to prevent overflows
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)