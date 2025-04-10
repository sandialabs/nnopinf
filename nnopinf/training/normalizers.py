import numpy as np
import os

class StandardNormalizer:
  '''
  Standard normalization class
  x_norm = (x - x_mean)/x_std
  '''
  def __init__(self,x):
    assert(x.ndim == 2)
    n_features = np.shape(x)[1]
    x_mean = np.mean(x)
    x_std = np.std(x)
    self.offset_value = x_mean
    self.scaling_value = x_std
    self.normalization_type='Standard'

  def get_offset_and_scaling_values(self):
    return self.offset_value,self.scaling_value

  def apply_scaling(self,x):
    x_normalized = (x - self.offset_value)/self.scaling_value
    return x_normalized

  def apply_inverse_scaling(self,x_normalized):
    x = x_normalized*self.scaling_value + self.offset_value
    return x


class MaxAbsNormalizer:
  '''
  MaxAbs normalization class
  x_norm = (x)/np.amax(np.abs(x_std))
  '''
  def __init__(self,x):
    assert(x.ndim == 2)
    n_features = np.shape(x)[1]
    self.scaling_value = np.zeros(n_features)
    self.offset_value = np.zeros(n_features)
    self.scaling_value[:] = np.amax(np.abs(x))
    self.normalization_type='MaxAbs'

  def get_offset_and_scaling_values(self):
    return self.offset_value,self.scaling_value

  def apply_scaling(self,x):
    x_normalized = x/self.scaling_value
    return x_normalized

  def apply_inverse_scaling(self,x_normalized):
    x = x_normalized*self.scaling_value
    return x



class NoOpNormalizer:
  '''
  MinMax normalization class
  x_norm = (x)/np.amax(np.abs(x_std))
  '''
  def __init__(self,x):
    assert(x.ndim == 2)
    n_features = np.shape(x)[1]
    self.scaling_value = 0
    self.offset_value = 1
    self.normalization_type='None'

  def get_offset_and_scaling_values(self):
    return self.offset_value,self.scaling_value

  def apply_scaling(self,x):
    return x

  def apply_inverse_scaling(self,x_normalized):
    return x_normalized


