import numpy as np
import yaml
import argparse


if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file",required=True)
  args = parser.parse_args()
  with open(args.i) as f:
        input_yaml_base = yaml.safe_load(f)
 
  bounds_left = [2.3375,5625.5/1e3]
  bounds_right = [6.2,9.0]
  n_params = len(bounds_left)

  ##Training
  np.random.seed(2)
  samples_1 = np.linspace(bounds_left[0],bounds_right[0],4)
  samples_2 = np.linspace(bounds_left[1],bounds_right[1],4)
  samples_1,samples_2 = np.meshgrid(samples_1,samples_2)
  samples = np.append(samples_1.flatten()[:,None],samples_2.flatten()[:,None],axis=1)
  np.savetxt(input_yaml_base['parameter-training-file'],samples,header=str(n_params),comments='',delimiter=',')
 
  ##Testing
  np.random.seed(3)
  samples_1 = np.linspace(bounds_left[0],bounds_right[0],7)[1::2]
  samples_2 = np.linspace(bounds_left[1],bounds_right[1],7)[1::2]
  samples_1,samples_2 = np.meshgrid(samples_1,samples_2)
  samples = np.append(samples_1.flatten()[:,None],samples_2.flatten()[:,None],axis=1)
  np.savetxt(input_yaml_base['parameter-testing-file'],samples,header=str(n_params),comments='',delimiter=',')
  
