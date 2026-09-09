import numpy as np
import yaml
import argparse


if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file",required=True)
  args = parser.parse_args()
  with open(args.i) as f:
        input_yaml_base = yaml.safe_load(f)
  
  bounds_left = [0.0]
  bounds_right = [0.0]

  n_params = len(bounds_left)
  
  ##Training
  np.random.seed(1)
  n_training_samples = int( input_yaml_base['training-parameter-set-cardinality'] )
  samples = np.random.uniform(low=bounds_left,high=bounds_right,size=(n_training_samples,n_params))

  np.savetxt(input_yaml_base['parameter-training-file'],samples,header=str(n_params),comments='',delimiter=',')
  
  
  ##Testing
  np.random.seed(2)
  n_testing_samples = int( input_yaml_base['testing-parameter-set-cardinality'] )
  samples = np.random.uniform(low=bounds_left,high=bounds_right,size=(n_testing_samples,n_params))
  np.savetxt(input_yaml_base['parameter-testing-file'],samples,header=str(n_params),comments='',delimiter=',')
  
