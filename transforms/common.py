class Compose(object):
  """Composes several transforms together.
  Single arg -> can be used for transforms with a single arg
  """

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, sample):
    for t in self.transforms:
      sample = t(sample)
    return sample

class ComposeCustom(object):
  """Composes several transforms together.
  Unpacks the args -> can be used for transforms with multiple args
  """

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, *args):
    for t in self.transforms:
      args = t(*args)
    return args
