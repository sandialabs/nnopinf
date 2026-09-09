import numpy as np
import tqdm
import argparse
import yaml
import sys
from utilities import *
from joblib import Parallel, delayed
try:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  Nranks = comm.Get_size()
except:
  rank = 0
  Nranks = 1
import pickle
from tqdm import tqdm
import scipy.optimize
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
torch.set_default_dtype(torch.float64)
import os.path
from scipy.sparse.linalg import LinearOperator
import os
import opinf
import nnopinf
import torch
import nnopinf.operators as operators
import nnopinf.models as models
import nnopinf.training
from energy_preserving_opinf import (
    _EnergyPreservingFitWorkspace,
    interpolate_energy_preserving_models,
)
from opinf_integrators import get_model_integrator, predict_continuous_model

## Load in training data
axis_font = {'size':'20'}
global numTimeSteps
global numSamples
global numNodes


def set_initial_seed(ml_yaml, ensemble_number):
  """Seed training RNGs for one ensemble member and return the seed used."""
  initial_seed = int(ml_yaml.get('initial-seed', 0))
  effective_seed = initial_seed + int(ensemble_number)
  np.random.seed(effective_seed)
  torch.manual_seed(effective_seed)
  return effective_seed


def get_energy_preserving_integration(input_yaml, model_type):
  """Return the EP-OpInf selection integrator, defaulting to RK4."""
  return get_model_integrator(input_yaml,model_type)


#class dataClasss:
#  '''
#  Data class equipped with member vectors features and response
#  '''
#  def __init__(self,features,response,featureNormalizationType,responseNormalizationType,featuresScalings,responseScalings):
#    self.features = features
#    self.response = response
#    self.featureNormalizationType = featureNormalizationType
#    self.responseNormalizationType = responseNormalizationType
#    self.featuresScalings = featuresScalings 
#    self.responseScalings = responseScalings

class DataClass:
  '''
  Data class equipped with member vectors features and response
  '''
  def __init__(self,features,response,feature_normalizer,response_normalizer):
    self.features = features
    self.response = response
    self.feature_normalizer = feature_normalizer
    self.response_normalizer = response_normalizer


### function to filter, split, and normalize data after reading it in
def prepareDataForTraining(train_percent,features,mu,response,feature_normalization_type,response_normalization_type):
  ## split into test and training
  n_samples = features.shape[0] 
  val_percent = 1. - train_percent
 
  samples_array = np.array(range(0,n_samples),dtype='int')
  np.random.shuffle(samples_array)
  train_samples = samples_array[0:int(np.floor(train_percent*n_samples))]
  val_samples = samples_array[int(np.floor(train_percent*n_samples))::]
 
  features_train = features[train_samples]
  features_val = features[val_samples]

  response_train = response[train_samples]
  response_val = response[val_samples]

  mu_train = mu[train_samples]
  mu_val = mu[val_samples]
  ## Now normalize data
  if response_normalization_type == 'Standard':
    response_normalizer = StandardNormalizer(response_train)
  elif response_normalization_type == 'MaxAbs':
    response_normalizer = MaxAbsNormalizer(response_train)
  elif response_normalization_type == 'None':
    response_normalizer = NoOpNormalizer(response_train)
  else:
    print("Normalizer not supported")

  ## Now normalize data
  if feature_normalization_type == 'Standard':
    feature_normalizer = StandardNormalizer(features_train)
  elif feature_normalization_type == 'MaxAbs':
    feature_normalizer = MaxAbsNormalizer(features_train)
  elif feature_normalization_type == 'None':
    feature_normalizer = NoOpNormalizer(features_train)
  else:
    print("Normalizer not supported")

  features_train = feature_normalizer.apply_scaling(features_train)
  features_val = feature_normalizer.apply_scaling(features_val)

  features_train = np.append(features_train,mu_train,axis=1)
  features_val = np.append(features_val,mu_val,axis=1)


  response_train = response_normalizer.apply_scaling(response_train)
  response_val = response_normalizer.apply_scaling(response_val)

  # Create DataClasses
  training_data = DataClass(features_train,response_train,feature_normalizer,response_normalizer)
  validation_data = DataClass(features_val,response_val,feature_normalizer,response_normalizer)
  return training_data,validation_data

def opinf_predict_with_parameters(model,u_ic,params,times):
    t = times[0]
    dt = times[1] - times[0]
    et = times[-1]
    u = u_ic*1.
    rk4const = np.array([1./4.,1./3.,1./2.,1.])
    u_states = np.zeros((u_ic.size,0))
    while t <= et + dt/2:
      u0 = u*1.
      u_states = np.append(u_states,u[:,None],axis=1)
      for i in range(0,4):
          f = model.rhs(0.,params,u).flatten() 
          u = u0 + dt*rk4const[i]*f
      t += dt
    return u_states

def opinf_predict(model,u_ic,times):
    t = times[0]
    dt = times[1] - times[0]
    et = times[-1]
    u = u_ic*1.
    rk4const = np.array([1./4.,1./3.,1./2.,1.])
    u_states = np.zeros((u_ic.size,0))
    while t <= et + dt/2:
      u0 = u*1.
      u_states = np.append(u_states,u[:,None],axis=1)
      for i in range(0,4):
          f = model.rhs(0.,u).flatten() 
          u = u0 + dt*rk4const[i]*f
      t += dt
    return u_states


def advance_fit_with_parameters(rom,reduced_states,reduced_states_dot,params,times):
   regularization_parameters_to_try = np.logspace(-8,3,40)
   errors = np.zeros(regularization_parameters_to_try.size)
   counter = 0
   extend_window_ratio = 5
   times_extended = times*1
   dt = times[1] - times[0]
   for i in range(1,extend_window_ratio):
     time_window = times_extended[-1] + (times - times[0]) + dt 
     times_extended = np.append(times_extended,time_window)
   assert(np.allclose(times_extended[0:times.size],times))
   for regularization_parameter in tqdm(regularization_parameters_to_try):
     l2solver = opinf.lstsq.L2Solver(regularizer=regularization_parameter)
     rom.solver = l2solver
     rom.fit(parameters=params,states=reduced_states, ddts=reduced_states_dot)#, solver=l2solver) 
     error = 0.
     for j in range(0,len(params)):
       u0 = reduced_states[j][...,0]
       test_states = opinf_predict_with_parameters(rom,u0,params[j],times_extended)
       if (test_states.shape[1] != times_extended.size) or (np.any(np.isnan(test_states)) or  np.any(np.abs(test_states) > 1e5)):
         print(test_states.shape[1],times_extended.size)
         print('BLEW UP')
         error = 1e10
       else:
         error += np.linalg.norm(test_states[:,0:times.size] - reduced_states[j]) / np.linalg.norm(reduced_states[j])
     errors[counter] = error
     counter += 1
   optimal_case = np.nanargmin(errors)
   optimal_regularization_parameter = regularization_parameters_to_try[optimal_case]
   print('Best regularization parameter = ' + str(optimal_regularization_parameter))
   l2solver = opinf.lstsq.L2Solver(regularizer=optimal_regularization_parameter)
   rom.solver = l2solver
   rom.fit(parameters=params,states=reduced_states, ddts=reduced_states_dot)#, solver=l2solver) 
   return rom

def advance_fit_no_parameters(rom,reduced_states,reduced_states_dot,times):
   regularization_parameters_to_try = np.logspace(-8,3,40)
   errors = np.zeros(regularization_parameters_to_try.size)
   counter = 0
   extend_window_ratio = 5
   times_extended = times*1
   dt = times[1] - times[0]
   for i in range(1,extend_window_ratio):
     time_window = times_extended[-1] + (times - times[0]) + dt 
     times_extended = np.append(times_extended,time_window)
   assert(np.allclose(times_extended[0:times.size],times))
   for regularization_parameter in regularization_parameters_to_try:
     l2solver = opinf.lstsq.L2Solver(regularizer=regularization_parameter)
     rom.solver = l2solver
     rom.fit(states=reduced_states[0], ddts=reduced_states_dot[0])#, solver=l2solver) 
     u0 = reduced_states[0][...,0]
#     try:
     #test_states = rom.predict(u0,times_extended)
     test_states = opinf_predict(rom,u0,times_extended)

     if (test_states.shape[1] != times_extended.size) or (np.any(np.isnan(test_states)) or  np.any(np.abs(test_states) > 1e5)):
       print(test_states.shape[1],times_extended.size)
       print('BLEW UP')
       error = 1e10
     else:
       error = np.linalg.norm(test_states[:,0:times.size] - reduced_states[0]) / np.linalg.norm(reduced_states[0])
     #except:
     #    print('BLew UP')
     #    error = 1e10
     errors[counter] = error
     print(error,regularization_parameter)
     counter += 1
   np.savez('errors_' + str(u0.shape[0]) , errors=errors,regularization_parameters_to_try = regularization_parameters_to_try)
   optimal_case = np.nanargmin(errors)
   optimal_regularization_parameter = regularization_parameters_to_try[optimal_case]
   print('Best regularization parameter = ' + str(optimal_regularization_parameter))
   l2solver = opinf.lstsq.L2Solver(regularizer=optimal_regularization_parameter)
   rom.solver = l2solver
   rom.fit(states=reduced_states[0], ddts=reduced_states_dot[0])#, solver=l2solver) 
   return rom

def advance_fit_energy_preserving(reduced_states,reduced_states_dot,params,times,model_form="AH",solver='direct',integration='rk4',verbose=1):
   if verbose not in (0,1,2):
     raise ValueError("verbose must be 0, 1, or 2")
   integration = str(integration).strip().lower()
   if integration not in ('rk4','crank-nicolson','cn','implicit-midpoint'):
     raise ValueError(
         "integration must be 'rk4', 'crank-nicolson', 'cn', "
         "or 'implicit-midpoint'"
     )
   search_start = time.perf_counter()
   regularization_parameters_to_try = np.logspace(-8,3,40)
   errors = np.zeros(regularization_parameters_to_try.size)
   workspaces = [
       _EnergyPreservingFitWorkspace(states,ddts,model_form=model_form)
       for states,ddts in zip(reduced_states,reduced_states_dot)
   ]
   warm_starts = [None] * len(workspaces)
   best_error = np.inf
   best_models = None
   if verbose:
     print(
         f"[EP-OpInf search] form={model_form} datasets={len(workspaces)} "
         f"candidates={regularization_parameters_to_try.size} "
         f"integration={integration}"
     )
     for index,workspace in enumerate(workspaces):
       estimates = workspace.cost_estimate()
       print(
           f"[EP-OpInf search] dataset={index} "
           f"r={workspace.state_dimension} snapshots={workspace.states.shape[1]} "
           f"features={workspace.operator_dimension} "
           f"parameters={workspace.parameter_dimension} "
           f"prep={workspace.preparation_seconds:.3f}s "
           f"design={estimates['design_bytes'] / 2**20:.1f}MiB "
           f"direct-hessian~{estimates['hessian_csr_bytes'] / 2**20:.1f}MiB "
           f"kkt-dimension={estimates['kkt_dimension']} "
           f"selected={workspace._resolve_solver(solver)}"
       )
   extend_window_ratio = 5
   times_extended = times*1
   dt = times[1] - times[0]
   for i in range(1,extend_window_ratio):
     time_window = times_extended[-1] + (times - times[0]) + dt
     times_extended = np.append(times_extended,time_window)

   parameterized = params.shape[0] > 1
   for counter,regularization_parameter in enumerate(regularization_parameters_to_try):
     candidate_start = time.perf_counter()
     try:
       models = []
       for index,workspace in enumerate(workspaces):
         model,warm_start = workspace.fit_model(
             regularization_parameter,
             initial_guess=warm_starts[index],
             verbose=2 if verbose >= 2 else 0,
             solver=solver,
         )
         models.append(model)
         warm_starts[index] = warm_start
       fit_seconds = time.perf_counter() - candidate_start
       rom = (
           interpolate_energy_preserving_models(params,models,model_form=model_form)
           if parameterized else models[0]
       )
       error = 0.
       integration_start = time.perf_counter()
       for j in range(len(reduced_states)):
         u0 = reduced_states[j][...,0]
         if parameterized:
           test_states = predict_continuous_model(
               rom,u0,times_extended,parameter=params[j],integration=integration
           )
         else:
           test_states = predict_continuous_model(
               rom,u0,times_extended,integration=integration
           )
         if (test_states.shape[1] != times_extended.size
             or np.any(~np.isfinite(test_states))
             or np.any(np.abs(test_states) > 1e5)):
           error = 1e10
           break
         error += (np.linalg.norm(test_states[:,0:times.size] - reduced_states[j])
                   / np.linalg.norm(reduced_states[j]))
       integration_seconds = time.perf_counter() - integration_start
       errors[counter] = error
       if error < best_error:
         best_error = error
         best_models = models
       if verbose:
         solver_summaries = []
         for workspace in workspaces:
           info = workspace.last_solver_info
           #if info['solver'] == 'matrix_free':
           #  solver_summaries.append(
           #      f"matrix_free:{info['iterations']}it/istop={info['istop']}"
           #  )
           #else:
           solver_summaries.append('direct')
         print(
             f"[EP-OpInf search] candidate={counter + 1}/"
             f"{regularization_parameters_to_try.size} "
             f"lambda={regularization_parameter:.3e} "
             f"fit={fit_seconds:.3f}s integrate={integration_seconds:.3f}s "
             f"total={time.perf_counter() - candidate_start:.3f}s "
             f"error={error:.3e} solvers={','.join(solver_summaries)}"
         )
     except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
       errors[counter] = 1e10
       if verbose:
         print(
             f"[EP-OpInf search] candidate={counter + 1}/"
             f"{regularization_parameters_to_try.size} "
             f"lambda={regularization_parameter:.3e} "
             f"FAILED after {time.perf_counter() - candidate_start:.3f}s: "
             f"{type(exc).__name__}: {exc}"
         )

   optimal_case = np.nanargmin(errors)
   optimal_regularization_parameter = regularization_parameters_to_try[optimal_case]
   if errors[optimal_case] >= 1e10:
     raise RuntimeError("All EP-OpInf regularization candidates failed")
   print('Best EP-OpInf regularization parameter = ' + str(optimal_regularization_parameter))
   if verbose:
     print(
         f"[EP-OpInf search] completed in "
         f"{time.perf_counter() - search_start:.3f}s "
         f"best-error={errors[optimal_case]:.3e}"
     )
   if parameterized:
     return interpolate_energy_preserving_models(params,best_models,model_form=model_form)
   return best_models[0]

def buildOpInfModel(Phi,params,times,reducedStates,reducedStatesDot,forcing,info,
                    ml_yaml,integration_settings=None):
  ep_verbose = int(ml_yaml.get('energy-preserving-verbosity',1))
  if integration_settings is None:
    integration_settings = ml_yaml
  ep_integration = get_energy_preserving_integration(
      integration_settings,info['model_type']
  )
  if info['model_type'] == 'OpInf-EP-cAH':
    rom = advance_fit_energy_preserving(
        reducedStates,reducedStatesDot,params,times,model_form="cAH",
        integration=ep_integration,verbose=ep_verbose
    )
  elif info['model_type'] == 'OpInf-EP-AH':
    rom = advance_fit_energy_preserving(
        reducedStates,reducedStatesDot,params,times,model_form="AH",
        integration=ep_integration,verbose=ep_verbose
    )
  elif info['model_type'] == 'OpInf-EP-H':
    rom = advance_fit_energy_preserving(
        reducedStates,reducedStatesDot,params,times,model_form="H",
        integration=ep_integration,verbose=ep_verbose
    )
  elif info['model_type'] == 'OpInf-cAH':
    if params.shape[0] == 1:
      rom = opinf.models.ContinuousModel("cAH")
      rom = advance_fit_no_parameters(rom,reducedStates,reducedStatesDot,times) 
    else:
      rom = opinf.models.InterpolatedContinuousModel("cAH")
      rom = advance_fit_with_parameters(rom,reducedStates,reducedStatesDot,params,times)
      #rom.fit(parameters=params,states=reducedStates, ddts=reducedStatesDot, solver=1.e-2)
  elif info['model_type'] == 'OpInf-cA':
    if params.shape[0] == 1:
      rom = opinf.models.ContinuousModel("cA")
      rom = advance_fit_no_parameters(rom,reducedStates,reducedStatesDot,times) 
    else:
      rom = opinf.models.InterpolatedContinuousModel("cA")
      rom = advance_fit_with_parameters(rom,reducedStates,reducedStatesDot,params,times)

  elif info['model_type'] == 'OpInf-pcA':
    if params.shape[0] == 1:
      rom = opinf.models.ContinuousModel("cA")
      rom = advance_fit_no_parameters(rom,reducedStates,reducedStatesDot,times) 
    else:
      print('HERE')
      operators=[
            opinf.operators.AffineConstantOperator(1),
            opinf.operators.AffineLinearOperator(1),
      ]
      rom = opinf.models.ParametricContinuousModel(operators)
      #rom = opinf.models.ParametricContinuousModel("cA")
      rom = advance_fit_with_parameters(rom,reducedStates,reducedStatesDot,params,times)

  elif info['model_type'] == 'OpInf-AH':
    if params.shape[0] == 1:
      rom = opinf.models.ContinuousModel("AH")
      rom = advance_fit_no_parameters(rom,reducedStates,reducedStatesDot,times) 

    else:
      rom = opinf.models.InterpolatedContinuousModel("AH")
      rom = advance_fit_with_parameters(rom,reducedStates,reducedStatesDot,params,times)

  elif info['model_type'] == 'OpInf-A':
    if params.shape[0] == 1:
      rom = opinf.models.ContinuousModel("A")
      rom = advance_fit_no_parameters(rom,reducedStates,reducedStatesDot,times) 
    else:
      rom = opinf.models.InterpolatedContinuousModel("A")
      rom = advance_fit_with_parameters(rom,reducedStates,reducedStatesDot,params,times)


  modelDir = os.getcwd() + '/' +  ml_yaml['full-training-output-directory']
  modelName = info['model_type'] + '_K_' + str(Phi.shape[1]) + '_stop_time_' + str(info['stop_time'])# + '_sample_' + str(info['sample'])  
  rom.save(modelDir + '/' + modelName + '_opinfrom.h5',overwrite=True)

def trainNeuralNetwork(model,modelDir,modelName,trainingData,validationData,n_epochs,ml_yaml):  
  if (os.path.isdir(modelDir) == False):
    os.mkdir(modelDir)
  device = 'cpu'
  model.to(device)
 
  nFeatures = trainingData.features.shape[1]
  nResponse = trainingData.response.shape[1]
  trainDataTorch = np.float64(np.append(trainingData.features,trainingData.response,axis=1))
  batch_size = int( ml_yaml['batch-size'] )
  trainLoader = torch.utils.data.DataLoader(trainDataTorch, batch_size=batch_size)
  
  valDataTorch = np.float64(np.append(validationData.features,validationData.response,axis=1))
  valLoader = torch.utils.data.DataLoader(valDataTorch, batch_size=batch_size)
  
  #Loss function
  def my_criterion(y,yhat):
    loss_mse = torch.mean( (y - yhat)**2 ) / torch.mean(y**2)
    #loss_mse = torch.mean( torch.abs(y - yhat)  / torch.abs(y) )
    #loss_mse = torch.mean( ( (y - yhat)/torch.abs(y + 1e-3) )**2 )
    return loss_mse

  #Optimizer
  learning_rate = ml_yaml['learning-rate']
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=ml_yaml['weight-decay'])
  lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=ml_yaml['lr-decay'])
 
  #Epochs
  train_loss_hist = np.zeros(0)
  val_loss_hist = np.zeros(0)
  
  t0 = time.time()
  epoch = 1
  while (epoch < n_epochs + 1):
  
      # monitor training loss
      train_loss = 0.0
      #Training
      nSamples = 0
      for data in trainLoader:
          data_d = data.to(device,dtype=torch.float64)
          inputs = data_d[:,0:nFeatures]
          y = data_d[:,nFeatures::]
          optimizer.zero_grad()
          yhat = model(inputs)
          loss = my_criterion(y,yhat)
          loss.backward()
          optimizer.step()
          train_loss += loss.item()*inputs.size(0)
          nSamples += inputs.size(0)


      train_loss = train_loss/nSamples
      train_loss_hist = np.append(train_loss_hist,train_loss)

  
      # monitor validation loss
      val_loss = 0.0
      #Training
      nSamples = 0
      for data in valLoader:
          data_d = data.to(device,dtype=torch.float64)
          inputs = data_d[:,0:nFeatures]
          y = data_d[:,nFeatures::]
          yhat = model.forward(inputs)
          loss = my_criterion(y,yhat)
          val_loss += loss.item()*inputs.size(0)
          nSamples += inputs.size(0)
  
      val_loss = val_loss/nSamples
      val_loss_hist = np.append(val_loss_hist,val_loss)

      lr_scheduler.step()
      lr = lr_scheduler.get_last_lr()[0]      
  
      if ml_yaml['print-training-output']:
        print('Epoch: {} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}'.format(epoch, train_loss,val_loss,lr))
        print("{:3d}       {:0.6f}        {:0.6f}     {:0.3e}".format(epoch, train_loss, val_loss, lr))
        print('Time: {:.6f}'.format(time.time() - t0))
  
      #if (epoch > 1000):
      #  val_loss_running_mean = np.mean(val_loss_hist[-400::])
      #  val_loss_running_mean_old = np.mean(val_loss_hist[-800:-400])
      #  if (val_loss_running_mean_old < val_loss_running_mean):
      #    print('MSE on validation set no longer decreasing, exiting training')
      #    epoch = 1e10
      epoch += 1
  #torch.save(model.state_dict(),modelDir + modelName)
  np.savez(modelDir + '/' + modelName + '_stats'  , train_loss=train_loss_hist,val_loss=val_loss_hist,walltime = time.time() - t0)
  torch.save(model, modelDir + '/' + modelName + '_model.pt')
  with open(modelDir + '/' + modelName + '_feature_normalizer.pickle', 'wb') as file:
    pickle.dump(trainingData.feature_normalizer, file) 
  with open(modelDir + '/' + modelName + '_response_normalizer.pickle', 'wb') as file:
    pickle.dump(trainingData.response_normalizer, file) 
  return model,trainingData.feature_normalizer,trainingData.response_normalizer

class buildANeuralNetworkModel:
  def __init__(self,Phi,features,mu,response,forcing,info,ml_yaml):

    self.scaling = 1.#np.linalg.norm(response) / np.linalg.norm(features)
    response /= self.scaling 

    numInput = features.shape[0] 
    numOutput = numInput 

    ## Model information
    numHiddenLayers = int(ml_yaml['num-hidden-layers'])
    numNeuronsPerLayer = int(ml_yaml['num-neurons-per-layer'])
  
    ## training information
    nEpochs = ml_yaml['num-epochs']
    modelDir = os.getcwd() + '/' +  ml_yaml['full-training-output-directory']
    makeRecursiveDirsIfNeeded(modelDir)
    modelName = info['model_type'] + '_K_' + str(Phi.shape[1]) + '_stop_time_' + str(info['stop_time']) + '_sample_' + str(info['sample'])  
    nFeatures = features.shape[0] + mu.shape[0] 
    nOutputs = response.shape[0]
    features = features.transpose()
    mu = mu.transpose()
    response = response.transpose()
    featureNormalizationType = 'MaxAbs'
    responseNormalizationType = 'MaxAbs'
    trainingData,validationData = prepareDataForTraining(0.8,features,mu,response,featureNormalizationType,responseNormalizationType) 
    model_type = info['model_type']
    if info['model_type'] == 'SS-SPD':
      model = MLPForSkewAndSpdOperator(numHiddenLayers,numNeuronsPerLayer,nFeatures,nOutputs,forcing)
    if info['model_type'] == 'Matrix':
      model = MLPForMatrixOperator(numHiddenLayers,numNeuronsPerLayer,nFeatures,nOutputs,forcing)
    if model_type == 'NN':
      model = MLP(numHiddenLayers,numNeuronsPerLayer,nFeatures,nOutputs,forcing)

    print("HERE",model_type)
    model,feature_normalizer,response_normalizer = trainNeuralNetwork(model,modelDir,modelName,trainingData,validationData,nEpochs,ml_yaml)
    #np.savez(modelDir + '/' + modelName + '_basis.npz',Phi=Phi) 
    self.model = model
    self.feature_normalizer = feature_normalizer
    self.response_normalizer = response_normalizer


def buildLagrangianOpInfModel(Phi,states,times,info,ml_yaml,modelDir,modelName):
  """Build the paper's two-stage LOpInf or LOpInf-SpML model."""
  if ml_yaml.get('parameterized',False):
    raise ValueError("LOpInf and LOpInf-SpML currently support nonparametric systems only")
  reduced_trajectories = np.einsum('ij,jtk->itk',Phi.transpose(),states)
  reduced_states = []
  reduced_velocities = []
  reduced_accelerations = []
  for trajectory in range(reduced_trajectories.shape[-1]):
    q,v,a,_ = nnopinf.training.eighth_order_time_derivatives(
        reduced_trajectories[...,trajectory],times
    )
    reduced_states.append(q)
    reduced_velocities.append(v)
    reduced_accelerations.append(a)
  q = np.concatenate(reduced_states,axis=1)
  v = np.concatenate(reduced_velocities,axis=1)
  a = np.concatenate(reduced_accelerations,axis=1)
  conservative = bool(ml_yaml.get('lagrangian-conservative',False))
  linear_model = nnopinf.training.fit_lagrangian_opinf(
      q,v,a,
      conservative=conservative,
      spd_epsilon=float(ml_yaml.get('lagrangian-spd-epsilon',1.e-8)),
      solver=ml_yaml.get('lagrangian-cvxpy-solver',None),
  )
  model = linear_model
  histories = {'train_loss':np.zeros(0),'validation_loss':np.zeros(0)}
  if info['model_type'] == 'LOpInf-SpML':
    widths = ml_yaml.get('lagrangian-hidden-layer-widths',None)
    if widths is None:
      widths = [Phi.shape[1]] * int(ml_yaml.get('num-hidden-layers',2))
    model = nnopinf.training.build_lagrangian_spml_model(
        linear_model,
        hidden_layer_widths=tuple(widths),
        maximum_polynomial_degree=int(ml_yaml.get('lagrangian-polynomial-degree',4)),
        num_subnetworks=int(ml_yaml.get('lagrangian-num-subnetworks',1)),
        learn_mass_correction=bool(ml_yaml.get('lagrangian-mass-correction',True)),
        nonlinear_potential=bool(ml_yaml.get('lagrangian-nonlinear-potential',True)),
        nonlinear_dissipation=(
            False if conservative
            else bool(ml_yaml.get('lagrangian-nonlinear-dissipation',True))
        ),
    )
    settings = {
        'num-epochs':int(ml_yaml.get('num-epochs',1000)),
        'batch-size':int(ml_yaml.get('batch-size',250)),
        'learning-rate':float(ml_yaml.get('learning-rate',1.e-4)),
        'lr-decay':float(ml_yaml.get('lr-decay',1.0)),
        'weight-decay':float(ml_yaml.get('weight-decay',0.0)),
        'validation-fraction':float(ml_yaml.get('validation-fraction',0.2)),
        'positivity-penalty-weight':float(ml_yaml.get('lagrangian-positivity-penalty-weight',1.0)),
        'seed':int(info['effective_seed']),
    }
    histories = nnopinf.training.train_lagrangian_spml(model,q,v,a,settings)
  torch.save(model,modelDir + '/' + modelName + '_sample_' + str(info['sample']) + '.pt')
  np.savez(
      modelDir + '/' + modelName + '_sample_' + str(info['sample']) + '_stats',
      train_loss=histories['train_loss'],
      val_loss=histories['validation_loss'],
      stiffness=model.stiffness_.detach().numpy(),
      damping=model.damping_.detach().numpy(),
  )
  return model


def buildModel(info,input_yaml_base):
  input_yaml = input_yaml_base['machine-learning']
  info['effective_seed'] = set_initial_seed(input_yaml, info['sample'])
  path_to_fom_sol = input_yaml_base['output-directory'] + '/' + input_yaml_base['fom']['fom-output-directory'] + '/' 

  skip = input_yaml['skip']
  data = np.load(path_to_fom_sol + 'fom_snapshots_training.npz')

  
  stop_index = np.argmin(np.abs(data['t'] - info['stop_time']))
  states = data['u'][:,0:stop_index:skip] #of shape N x Nt 

  #noise = np.random.normal(size=states.shape)
  #noise *= np.mean(np.abs(states))*0.05
  #states += noise
  times = data['t'][0:stop_index:skip]

  dt = times[2] - times[1]

  statesDot = (states[:,2::] - states[:,0:-2] )/ (2.*dt)
  #statesDot[:,1:-1] = (-states[:,4:] + 8*states[:,3:-1] - 8*states[:,1:-3] + states[:,0:-4]) / (12.*dt)

  mu = data['params'][:,0:stop_index:skip][:,1:-1]
  n_params = mu.shape[0]
  mu  = np.reshape(mu,(mu.shape[0],np.prod(mu.shape[1::]))) 
  print('HERE')
  modelDir = os.getcwd() + '/' +  input_yaml['full-training-output-directory']
  modelName = info['model_type'] + '_K_' + str(info['K']) + '_stop_time_' + str(info['stop_time'])# + '_sample_' + str(info['sample']) 
  makeRecursiveDirsIfNeeded(modelDir)
  U,S,V = np.linalg.svd(np.reshape(states,(states.shape[0],np.prod(states.shape[1::]))),full_matrices=False)
  print('Did SVD!')
  relativeEnergy = np.cumsum(S**2) / np.sum(S**2)
  K = info['K']
  Phi = U[:,0:K]

  if info['model_type'] in ('LOpInf','LOpInf-SpML'):
    buildLagrangianOpInfModel(
        Phi,states,times,info,input_yaml,modelDir,modelName
    )
    np.savez(modelDir + '/' + modelName + '_basis.npz',Phi=Phi)
    return
  reducedStates = np.einsum('ij,jkl->ikl',Phi.transpose() , states[:,1:-1])



  forcing = Phi.transpose() @ data['forcing'] 
  reducedStatesDot = np.einsum('ij,jkl->ikl',Phi.transpose() , statesDot) 

  if info['model_type'][0:5] == 'OpInf':
    parameter_means = np.mean(mu,axis=1)
    parameter_std = np.std(mu,axis=1)
    mu = (mu - parameter_means[:,None])/(parameter_std[:,None] + 1e-2)
    reducedStatesList = []
    reducedStatesDotList = []
    params = data['params'][:,0,:]
    for i in range(0,reducedStates.shape[-1]):
       reducedStatesList.append(reducedStates[...,i]) 
       reducedStatesDotList.append(reducedStatesDot[...,i]) 
    opInfModel = buildOpInfModel(
        Phi,params.transpose(),times[1:-1],reducedStatesList,
        reducedStatesDotList,forcing,info,input_yaml,
        integration_settings=input_yaml_base
    )
  elif info['model_type'][0:4] == 'NNOP':
    parameter_means = np.zeros(mu.shape)
    parameter_std = np.ones(mu.shape)

    reducedStates = np.reshape(reducedStates,(reducedStates.shape[0],np.prod(reducedStates.shape[1::])))  
    reducedStatesDot = np.reshape(reducedStatesDot,(reducedStatesDot.shape[0],np.prod(reducedStatesDot.shape[1::])))  
    neuralNetworkModel = train_model(input_yaml,info,reducedStates,reducedStatesDot,mu)
    
  elif info['model_type'] != 'ROM':
    parameter_means = np.mean(mu,axis=1)
    parameter_std = np.std(mu,axis=1)
    mu = (mu - parameter_means[:,None])/(parameter_std[:,None] + 1e-2)

    reducedStates = np.reshape(reducedStates,(reducedStates.shape[0],np.prod(reducedStates.shape[1::])))  
    #reducedStates = np.append(reducedStates,mu,axis=0)
    reducedStatesDot = np.reshape(reducedStatesDot,(reducedStatesDot.shape[0],np.prod(reducedStatesDot.shape[1::])))  
    neuralNetworkModel = buildANeuralNetworkModel(Phi,reducedStates,mu,reducedStatesDot,forcing,info,input_yaml)

  if info['model_type'] != 'ROM':
    np.savez(modelDir + '/' + modelName + '_basis.npz',Phi=Phi) 
    np.savez(modelDir + '/' + modelName + '_parameter_scalings',parameter_means=parameter_means,parameter_std=parameter_std)
  else:
    np.savez(modelDir + '/' + modelName + '_basis.npz',Phi=Phi) 

def train_model(input_yaml,info,uhat,uhat_dot,params=None,initialization_model=None):
    ensemble_id = info['sample']
    ml_info = input_yaml 
    output_dir = os.getcwd() + '/' +  ml_info['full-training-output-directory']
    # Load in snapshots
    cur_dir = os.getcwd()
    # Set values = 0 if DOFs are fixed

    rom_dim = uhat.shape[0] 

    n_hidden_layers = ml_info['num-hidden-layers'] 
    n_neurons_per_layer = rom_dim
    n_inputs = rom_dim 
    n_outputs = rom_dim
    n_params = np.shape(params)[0]
    ## Design operators for the state
    xvar = nnopinf.variables.Variable(size=rom_dim,name='x',normalization_strategy='MaxAbs')
    muvar = nnopinf.variables.Variable(size=n_params,name='mu',normalization_strategy='MaxAbs')
    target = nnopinf.variables.Variable(size=rom_dim,name='y',normalization_strategy='MaxAbs')
    if ml_info['parameterized'] == True: 
      NpdMlp = nnopinf.operators.SpdOperator(acts_on=xvar,depends_on=(xvar,muvar),n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer,positive=False)
      SkewMlp = nnopinf.operators.SkewOperator(acts_on=xvar,depends_on=(xvar,muvar),n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer )
      StandardMlp = nnopinf.operators.StandardOperator(depends_on=(xvar,muvar),n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer,n_outputs=rom_dim)

    else: 
      NpdMlp = nnopinf.operators.SpdOperator(acts_on=xvar,depends_on=(xvar,),n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer,positive=False)
      SkewMlp = nnopinf.operators.SkewOperator(acts_on=xvar,depends_on=(xvar,),n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer)
      StandardMlp = nnopinf.operators.StandardOperator(depends_on=(xvar,),n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer,n_outputs=rom_dim)
    VectorOffset = nnopinf.operators.VectorOffsetOperator(n_outputs=rom_dim)
    #MatrixMlp = operators.MatrixOperator(n_hidden_layers,n_neurons_per_layer,n_inputs + n_params,(n_outputs,n_outputs))
    #StandardMlp = operators.StandardOperator(n_hidden_layers,n_neurons_per_layer,n_inputs + n_params,n_outputs)
    #LinearNpdMlp = operators.SpdLinearOperator(n_inputs,n_params,n_outputs,positive=False)
    #LinearSkewMlp = operators.SkewLinearOperator(n_inputs,n_params,n_outputs)
    #LinearStandardMlp = operators.StandardLinearOperator(n_inputs,n_params,n_outputs)
    #NdMlp = operators.CompositeOperator([NpdMlp,SkewMlp])
    #LinearNdMlp = operators.CompositeOperator([LinearNpdMlp,LinearSkewMlp])

    #LinearForcingOperator = operators.StandardOperator(0,0,0,n_outputs)
    #ForcingOperator = operators.StandardOperator(0,0,n_params,n_outputs)
    #standard_operator = models.WrappedOperatorForModel(operator=StandardMlp,inputs=("x","mu",),name="standard-" + str(ensemble_id))

    #matrix_operator =  models.WrappedOperatorForModel(operator=MatrixMlp,inputs=("x","mu",),name="ts-" + str(ensemble_id))

    #npd_operator =  models.WrappedOperatorForModel(operator=NpdMlp,inputs=("x","mu",),name="npd-" + str(ensemble_id))
    #nd_operator =  models.WrappedOperatorForModel(operator=NdMlp,inputs=("x","mu",),name="nd-" + str(ensemble_id))
    #lin_operator =  models.WrappedOperatorForModel(operator=LinearStandardMlp,inputs=("x","mu",),name="linear-" + str(ensemble_id))
    #lin_nd_operator =  models.WrappedOperatorForModel(operator=LinearNdMlp,inputs=("x","mu",),name="linear-nd-" + str(ensemble_id))
#
#    linear_forcing_operator =  models.WrappedOperatorForModel(operator=LinearForcingOperator,inputs=("null",),name="linear-forcing-" + str(ensemble_id))
#    forcing_operator =  models.WrappedOperatorForModel(operator=ForcingOperator,inputs=("mu",),name="forcing-" + str(ensemble_id))

    if info['model_type'] == 'NNOPINF-TS': 
      my_operators = []   
      my_operators.append(matrix_operator) 
    if info['model_type'] == 'NNOPINF-TS-f': 
      my_operators = []   
      my_operators.append(matrix_operator) 
      my_operators.append(forcing_operator) 
    if info['model_type'] == 'NNOPINF-SPD': 
      my_operators = []   
      my_operators.append(npd_operator) 
    if info['model_type'] == 'NNOPINF-SS':
      my_model = nnopinf.models.Model( [SkewMlp] )

    if info['model_type'] == 'NNOPINF-SPD-f':
      my_model = nnopinf.models.Model( [NpdMlp,VectorOffset] )
    if info['model_type'] == 'NNOPINF-PD':
      my_model = nnopinf.models.Model( [NpdMlp,SkewMlp] )
    if info['model_type'] == 'NNOPINF-LIN-PD-f':
      my_operators = []   
      my_operators.append(lin_nd_operator) 
      my_operators.append(linear_forcing_operator) 
    if info['model_type'] == 'NNOPINF-LIN-f':
      my_operators = []   
      my_operators.append(lin_operator) 
      my_operators.append(linear_forcing_operator) 
    if info['model_type'] == 'NNOPINF-PD-f':
      my_model = nnopinf.models.Model( [NpdMlp,SkewMlp,VectorOffset] )

    if info['model_type'] == 'NNOPINF-NN-f':
      my_operators = []   
      my_operators.append([StandardMlp]) 
      my_operators.append(forcing_operator) 
    if info['model_type'] == 'NNOPINF-NN':
      my_model = nnopinf.models.Model([StandardMlp]) 

    #my_model = models.OpInfModel( my_operators )
    if initialization_model is not None:
        my_model.hierarchical_update(initialization_model)

    #Construct training data
    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs(output_dir)

    training_settings = nnopinf.training.get_default_settings()
    training_settings['num-epochs'] =  ml_info['num-epochs']
    training_settings['learning-rate'] = ml_info['learning-rate']
    training_settings['lr-decay'] = ml_info['lr-decay']
    training_settings['weight-decay'] = ml_info['weight-decay']
    training_settings['optimizer'] = ml_info['optimizer']
    training_settings['LBFGS-acceleration'] = ml_info['LBFGS-acceleration']
    #training_settings['TR-NEWTON-acceleration'] = True#ml_info['LBFGS-acceleration']
    #training_settings['TR-NEWTON-acceleration-epoch-frequency'] = 250#ml_info['LBFGS-acceleration']
    #training_settings['TR-NEWTON-acceleration-iterations'] = 50#ml_info['LBFGS-acceleration']
    #training_settings['batch-size'] = 1000 
    training_settings['output-path'] = output_dir
    training_settings['model-name'] = info['model_type'] + '_K_' + str(K) + '_stop_time_' + str(info['stop_time']) + '_sample_' + str(ensemble_id)
    training_settings['print-training-output'] = False 
    #training_settings['GN-num-layers'] = 1#ml_info['GN-num-layers']
    #training_settings['GN-final-layer'] = False#ml_info['GN-final-layer']
    #training_settings['GN-final-layer-epoch-frequency'] = 1#ml_info['GN-final-layer-epoch-frequency']
    inputs = {}
    inputs['x'] = uhat.transpose()
    inputs['mu'] = params.transpose()
    inputs['null'] = np.zeros((uhat.transpose().shape[0],0)) 
    training_settings['x-normalization-strategy'] = 'MaxAbs'
    training_settings['mu-normalization-strategy'] = 'Abs'

    xvar.set_data(uhat.transpose())
    muvar.set_data(params.transpose())
    target.set_data(uhat_dot.transpose())
    variables = [xvar,muvar]
    nnopinf.training.train(my_model,variables=variables,y=target,training_settings=training_settings)
    #vals_to_save = {}
    #vals_to_save["basis"] = trial_space.get_basis() 
    #np.savez(output_dir + '/nn-opinf-basis',**vals_to_save)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file",required=True)
  args = parser.parse_args()
  with open(args.i) as f:
        input_yaml_base = yaml.safe_load(f)
  input_yaml = input_yaml_base['machine-learning']

  input_yaml['full-training-output-directory'] = input_yaml_base['output-directory'] + '/' + input_yaml['training-output-directory']
  KsToRun = np.array(input_yaml['reduced-basis-dimensions'],dtype='int')

  stop_times = np.array(input_yaml['training-stop-times'],dtype=float)
  model_types = input_yaml['model-types'] 
  if rank == 0:
    print('Training ML models')

  training_jobs = build_training_jobs(
      stop_times,model_types,KsToRun,input_yaml
  )
  n_models = len(training_jobs)
  runs_per_rank = int(np.ceil(n_models*1./Nranks))
  starting_rank = min(runs_per_rank * rank,n_models)
  ending_rank = min( runs_per_rank*(rank+1),n_models)
  for model_no in range(starting_rank,ending_rank):
    if rank == 0:
      print('Training ' + str(model_no) + ' of ' + str(ending_rank) + ' models',flush=True)
    stop_time,model_type,K,sample = training_jobs[model_no]
    info = {}
    info['model_type'] = model_type
    info['stop_time'] = stop_time
    info['sample'] = sample
    info['K'] = K
    buildModel(info,input_yaml_base)
