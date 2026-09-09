import numpy as np
import yaml
import argparse


if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file",required=True)
  args = parser.parse_args()
  with open(args.i) as f:
        input_yaml_base = yaml.safe_load(f)
 
  ## Parameters are:
  # sigma: forcing on RHS
  # nu: viscosity
  # theta: angle of wave speed  
  bounds_left = [2.3375,5625.5/1e3]
  bounds_right = [6.2,9.0]
  n_params = len(bounds_left)

  ##Training
  np.random.seed(2)
  n_training_samples = int( input_yaml_base['training-parameter-set-cardinality'] )
  samples = np.random.uniform(low=bounds_left,high=bounds_right,size=(n_training_samples,n_params))

  if input_yaml_base['sample-corners']:
    n_params = len(bounds_left)
    corners = np.zeros((n_params,2))
    for i in range(0,n_params):
      corners[i]  = np.array([bounds_left[i],bounds_right[i]])
  
    p1,p2 = np.meshgrid(corners[0],corners[1])
    p1 = p1.flatten()
    p2 = p2.flatten()
    corner_samples = np.append(p1[:,None],p2[:,None],axis=1)
    samples = np.append(corner_samples,samples,axis=0)
    samples = samples[:,0:n_training_samples]

  np.savetxt(input_yaml_base['parameter-training-file'],samples,header=str(n_params),comments='',delimiter=',')
 
  ##Testing
  np.random.seed(3)
  n_testing_samples = int( input_yaml_base['testing-parameter-set-cardinality'] )
  samples = np.random.uniform(low=bounds_left,high=bounds_right,size=(n_testing_samples,n_params))
  np.savetxt(input_yaml_base['parameter-testing-file'],samples,header=str(n_params),comments='',delimiter=',')
  
