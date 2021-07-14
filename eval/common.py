import numpy as np

from eval.sem_seg_2d import fast_hist, per_class_iu


class ConfMat:
  '''
  Confusion matrix that can be updated repeatedly
  and later give IoU, accuracy and the matrix itself
  '''
  def __init__(self, num_classes):
    self.num_classes = num_classes
    self._mat = np.zeros((self.num_classes, self.num_classes))

  def reset(self):
    self._mat *= 0

  @property
  def ious(self):
    return per_class_iu(self._mat)

  @property
  def accs(self):
    return self._mat.diagonal() / self._mat.sum(1) * 100

  @property
  def mat(self):
    return self._mat

  def update(self, preds, targets):
    '''
    preds, targets: torch Tensor b, x, y, z.. 
    '''
    self._mat += fast_hist(preds.cpu().numpy().flatten(), 
                          targets.cpu().numpy().flatten(), 
                          self.num_classes)