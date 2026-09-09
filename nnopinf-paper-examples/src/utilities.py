import numpy as np
import os
from matplotlib import pyplot as plt


def model_uses_samples(model_type):
    """Return whether a model family is trained as a stochastic ensemble."""
    model_type = str(model_type)
    return not (
        model_type == 'ROM'
        or model_type.startswith('OpInf')
        or model_type == 'LOpInf'
    )


def get_num_training_samples(ml_yaml):
    """Return the configured number of stochastic training members."""
    has_ensemble_count = 'num-ensembles' in ml_yaml
    has_ensemble_sizes = 'ensemble-sizes' in ml_yaml
    if has_ensemble_count != has_ensemble_sizes:
        raise ValueError(
            "'num-ensembles' and 'ensemble-sizes' must be specified together"
        )
    if not has_ensemble_count:
        samples = int(ml_yaml['num-samples'])
        if samples < 1:
            raise ValueError("'num-samples' must be positive")
        return samples
    num_ensembles = int(ml_yaml['num-ensembles'])
    ensemble_sizes = np.asarray(ml_yaml['ensemble-sizes'], dtype=int)
    if num_ensembles < 1:
        raise ValueError("'num-ensembles' must be positive")
    if ensemble_sizes.size == 0 or np.any(ensemble_sizes < 1):
        raise ValueError("'ensemble-sizes' must contain positive integers")
    return num_ensembles * int(np.max(ensemble_sizes))


def model_sample_indices(model_type, ml_yaml):
    """Return sample indices for one model, using zero for deterministic models."""
    if not model_uses_samples(model_type):
        return np.array([0], dtype=int)
    return np.arange(get_num_training_samples(ml_yaml), dtype=int)


def build_training_jobs(stop_times, model_types, basis_dimensions, ml_yaml):
    """Build training jobs with sample expansion only for stochastic models."""
    jobs = []
    for stop_time in stop_times:
        for model_type in model_types:
            samples = model_sample_indices(model_type, ml_yaml)
            for basis_dimension in basis_dimensions:
                for sample in samples:
                    jobs.append(
                        (
                            float(stop_time),
                            str(model_type),
                            int(basis_dimension),
                            int(sample),
                        )
                    )
    return jobs


def mesh_grid_flatten(*args):
    # Create the meshgrid for the provided arguments
    grids = np.meshgrid(*args)

    # Flatten each grid and return as a tuple
    flattened_grids = [grid.flatten() for grid in grids]
    return tuple(flattened_grids)

#def mesh_grid_flatten_three_arg(arg1,arg2,arg3):
#   arg1,arg2,arg3 = np.meshgrid(arg1,arg2,arg3)
#   arg1 = arg1.flatten()
#   arg2 = arg2.flatten()
#   arg3 = arg3.flatten()
#   return arg1,arg2,arg3
#
#def mesh_grid_flatten(arg1,arg2,arg3,arg4):
#   arg1,arg2,arg3,arg4 = np.meshgrid(arg1,arg2,arg3,arg4)
#   arg1 = arg1.flatten()
#   arg2 = arg2.flatten()
#   arg3 = arg3.flatten()
#   arg4 = arg4.flatten()
#   return arg1,arg2,arg3,arg4

def parameter_reader(filename):
  f = open(filename, "r")
  n_params = int(f.readline()[0:-1])
  f = np.genfromtxt(filename,skip_header=1,delimiter=',')
  if f.ndim == 0:
    n_samples = 1
    f = np.array([f])
    f = f[None]

  if f.ndim == 1:
    n_samples = 1
    f = f[None]

  return f

def makeRecursiveDirsIfNeeded(saveDir):
  # delete double slashes
  saveDir = saveDir.replace('//','/')
  workingDir = ''
  #if saveDir[0] != '/':
  #  start_string = 0
  #else:
  #  start_string = 1 
  for directory in saveDir.split('/')[0:-1]:
    workingDir += directory
    if os.path.isdir(workingDir + '/'):
      pass
    else:
      try:
        os.mkdir(workingDir)
      except:
        print('Cant make working directory')
    workingDir += '/'

#====================


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
    self.offset_value = 0
    self.scaling_value = np.amax(np.abs(x))
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




'''
Functions to compute and apply scalings
'''

def normalizeStandard(x):
  '''
  function to perform standard normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  nFeatures = np.shape(x)[1]
  xScalings = np.zeros((2,nFeatures))
  xMean = np.mean(x,axis=0)
  xStd = np.std(x,axis=0)

  xScalings[0] = xMean
  xScalings[1] = xStd

  xNorm = (x - xMean)/xStd
  return xNorm,xScalings


def normalizeMinMax(x):
  '''
  function to perform min max normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  nFeatures = np.shape(x)[1]
  xScalings = np.zeros((2,nFeatures))
  xMin = np.min(x,axis=0)
  xMax = np.max(x,axis=0)

  xScalings[0] = xMin
  xScalings[1] = xMax - xMin

  xNorm = (x - xMin)/xScalings[1]
  return xNorm,xScalings


def normalizeLogSign(x):
  '''
  function to perform a log normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  nFeatures = np.shape(x)[1]
  xScalings = np.zeros((2,nFeatures)) #not used here, but keep for consistency
  xNorm = np.sign(x)*np.log10(np.abs(x))
  return xNorm,xScalings


def normalizeSmoothLogSign(x,C=-5):
  '''
  function to perform a smooth log normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  nFeatures = np.shape(x)[1]
  xScalings = np.zeros((2,nFeatures)) #not used here, but keep for consistency
  xNorm = np.sign(x)*np.log(1 + np.abs(x)/10**C) 
  return xNorm,xScalings


def normalizeNone(x):
  '''
  function to perform no normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  nFeatures = np.shape(x)[1]
  xScalings = np.zeros((2,nFeatures)) #not used here, but keep for consistency
  xNorm = x 
  return xNorm,xScalings


#==========================
'''
Functions to apply scalings
'''

def applyNormalizeStandard(x,xScalings):
  '''
  function to apply standard normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  xNorm = (x - xScalings[0])/xScalings[1]
  return xNorm


def applyNormalizeMinMax(x,xScalings):
  '''
  function to apply  min max normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  xNorm = (x - xScalings[0])/xScalings[1]
  return xNorm


def applyNormalizeLogSign(x,xScalings=None):
  '''
  function to apply a log normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  xNorm = np.sign(x)*np.log10(np.abs(x))
  return xNorm


def applyNormalizeSmoothLogSign(x,xScalings=None,C=-5):
  '''
  function to apply a smooth log normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  xNorm = np.sign(x)*np.log(1 + np.abs(x)/10**C) 
  return xNorm


def applyNormalizeNone(x,xScalings=None):
  '''
  function to apply no normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  xNorm = x 
  return xNorm



#==========================
'''
Functions to apply inverse scalings
'''

def applyInverseNormalizeStandard(xNorm,xScalings):
  '''
  function to apply standard normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(xNorm.ndim == 2)
  x = xNorm*xScalings[1] + xScalings[0]
  return x


def applyInverseNormalizeMinMax(xNorm,xScalings):
  '''
  function to apply  min max normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(xNorm.ndim == 2)
  x = xNorm*xScalings[1] + xScalings[0]
  return x



def applyInverseNormalizeNone(xNorm,xScalings=None):
  '''
  function to apply no normalization.
  We assume first axis is number of batches, second axes is features
  ''' 
  assert(x.ndim == 2)
  return xNorm
