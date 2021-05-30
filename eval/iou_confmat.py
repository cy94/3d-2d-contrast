'''
adapted from: 
https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/functional/classification/iou.py#L47-L111
returns both IOU and confmat
'''

from typing import Optional

import torch
from torch import Tensor

import torchmetrics as tmetrics
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType
from torchmetrics.utilities.data import get_num_classes
from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_update

from torchmetrics.functional.classification.iou import _iou_from_confmat

def iou_confmat(
    preds: Tensor,
    target: Tensor,
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    reduction: str = 'elementwise_mean',
) -> Tensor:
    r"""
    Computes `Intersection over union, or Jaccard index calculation <https://en.wikipedia.org/wiki/Jaccard_index>`_:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Where: :math:`A` and :math:`B` are both tensors of the same size,
    containing integer class values. They may be subject to conversion from
    input data (see description below).

    Note that it is different from box IoU.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If pred has an extra dimension as in the case of multi-class scores we
    perform an argmax on ``dim=1``.

    Args:
        preds: tensor containing predictions from model (probabilities, or labels) with shape ``[N, d1, d2, ...]``
        target: tensor containing ground truth labels with shape ``[N, d1, d2, ...]``
        ignore_index: optional int specifying a target class to ignore. If given,
            this class index does not contribute to the returned score, regardless
            of reduction method. Has no effect if given an int that is not in the
            range [0, num_classes-1], where num_classes is either given or derived
            from pred and target. By default, no index is ignored, and all classes are used.
        absent_score: score to use for an individual class, if no instances of
            the class index were present in `pred` AND no instances of the class
            index were present in `target`. For example, if we have 3 classes,
            [0, 0] for `pred`, and [0, 2] for `target`, then class 1 would be
            assigned the `absent_score`.
        threshold:
            Threshold value for binary or multi-label probabilities. default: 0.5
        num_classes:
            Optionally specify the number of classes
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        IoU score : Tensor containing single value if reduction is
        'elementwise_mean', or number of classes if reduction is 'none'

    Example:
        >>> from torchmetrics.functional import iou
        >>> target = torch.randint(0, 2, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> iou(pred, target)
        tensor(0.9660)
    """

    num_classes = get_num_classes(preds=preds, target=target, num_classes=num_classes)
    confmat = _confusion_matrix_update(preds, target, num_classes, threshold)
    return _iou_from_confmat(confmat, num_classes, ignore_index, absent_score, reduction), confmat