import numpy as np

def fast_hist(pred, label, n):
  k = (label >= 0) & (label < n)
  return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)

def per_class_iu(hist):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

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