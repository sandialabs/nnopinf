import numpy as np
import types
import sys
import time
from utilities import *
import pickle
import torch
import opinf
import nnopinf
import scipy.optimize
from opinf_integrators import first_order_step, get_model_integrator

_EGREGIOUS_ROM_VALUE = 1e10

try:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  Nranks = comm.Get_size()
except:
  rank = 0
  Nranks = 1

class generic_rom:
  def __init__(self,myFom):
    self.myFom = myFom

 
  def solve(self,model_type,model_name,params,input_yaml_base,u0):
    rom_input_yaml = input_yaml_base['roms']
    fom_input_yaml = input_yaml_base['fom']
    ml_input_yaml = input_yaml_base['machine-learning']
    integration = get_model_integrator(input_yaml_base,model_type)
    print(f"Integrating {model_type} with {integration}")
    if model_type[0:4] == 'NNOP':
      Phi = np.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_basis.npz')['Phi']
    elif model_type != 'ROM':
      Phi = np.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_basis.npz')['Phi']
      parameter_scalings = np.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_parameter_scalings.npz')
      mu_std = parameter_scalings['parameter_std']
      mu_mean = parameter_scalings['parameter_means']
    else:
      print(model_name)
      Phi = np.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/ROM'  + model_name[3::] + '_basis.npz')['Phi']

    K = Phi.shape[1]
    uhat = np.zeros(K) 
    myFom = self.myFom 
    if model_type == 'ROM':
      u_snapshots,t_snapshots = solve_galerkin_rom(Phi,myFom,params,fom_input_yaml,u0,integration)
    else:
      if model_type == 'OpInf-EP-cAH':
        try:
          model = opinf.models.InterpolatedContinuousModel("cAH")
          model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
          parametric = True
        except:
          try:
            model = opinf.models.ContinuousModel("cAH")
            model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
            parametric = False
          except:
            print("Failed to load model")
            sys.exit()
        u_snapshots,t_snapshots = solve_opinf_rom(Phi,model,params,fom_input_yaml,u0,parametric,integration)

      elif model_type == 'OpInf-EP-H':
        try:
          model = opinf.models.InterpolatedContinuousModel("H")
          model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
          parametric = True
        except:
          try:
            model = opinf.models.ContinuousModel("H")
            model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
            parametric = False
          except:
            print("Failed to load model")
            sys.exit()
        u_snapshots,t_snapshots = solve_opinf_rom(Phi,model,params,fom_input_yaml,u0,parametric,integration)

      elif model_type == 'OpInf-EP-AH':
        try:
          model = opinf.models.InterpolatedContinuousModel("AH")
          model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
          parametric = True
        except:
          try:
            model = opinf.models.ContinuousModel("AH")
            model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
            parametric = False
          except:
            print("Failed to load model")
            sys.exit()
        u_snapshots,t_snapshots = solve_opinf_rom(Phi,model,params,fom_input_yaml,u0,parametric,integration)

      elif model_type == 'OpInf-cA':
        try:
          ## Attempt to load an interpolated model
          model = opinf.models.InterpolatedContinuousModel("cA")
          model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
          parametric=True

        except:
            #If not, try and load a non parametric model
          try:
            model = opinf.models.ContinuousModel("cA")
            model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
            parametric = False
          except:
            print("Failed to load model")
            sys.exit()

        u_snapshots,t_snapshots = solve_opinf_rom(Phi,model,params,fom_input_yaml,u0,parametric,integration)

      elif model_type == 'OpInf-cAH':
        try:
          model = opinf.models.InterpolatedContinuousModel("cAH")
          model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
          parametric = True
        except:
          try:
            model = opinf.models.ContinuousModel("cAH")
            model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
            parametric = False
          except:
            print("Failed to load model")
            sys.exit()
        u_snapshots,t_snapshots = solve_opinf_rom(Phi,model,params,fom_input_yaml,u0,parametric,integration)
      elif model_type == 'OpInf-A':
        try:
          ## Attempt to load an interpolated model
          model = opinf.models.InterpolatedContinuousModel("A")
          model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
          parametric = True
        except:
            #If not, try and load a non parametric model
          try:
            parametric = False
            model = opinf.models.ContinuousModel("A")
            model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
          except:
            print("Failed to load model")
            sys.exit()
        u_snapshots,t_snapshots = solve_opinf_rom(Phi,model,params,fom_input_yaml,u0,parametric,integration)

      elif model_type == 'OpInf-AH':
        try:
          model = opinf.models.InterpolatedContinuousModel("AH")
          model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
          parametric = True
        except:
          try:
            parametric = False
            model = opinf.models.ContinuousModel("AH")
            model = model.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_opinfrom.h5')
          except:
            parametric = False
            print("Failed to load model")
            sys.exit()
        u_snapshots,t_snapshots = solve_opinf_rom(Phi,model,params,fom_input_yaml,u0,parametric,integration)
        

      elif model_type == 'entropy':
        model = torch.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_model.pt')

        with open(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_feature_normalizer.pickle', 'rb') as file:
          feature_normalizer = pickle.load(file)
        with open(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_response_normalizer.pickle', 'rb') as file:
          response_normalizer = pickle.load(file)

        u_snapshots,t_snapshots = solve_entropy_ml_rom(self,Phi,model,params,feature_normalizer,response_normalizer,mu_mean,mu_std,fom_input_yaml,u0,integration) 

      elif model_type[0:4] == 'NNOP':
        models = []
        for i in model_sample_indices(model_type,ml_input_yaml):
          checkpoint = input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_sample_' + str(i) + '.pt'
          # NNOpInf saves the complete Model object rather than a state_dict.
          # These checkpoints are generated locally and therefore trusted.
          models.append(torch.load(checkpoint,weights_only=False))
        u_snapshots,t_snapshots = solve_nnopinf_rom(Phi,models,params,fom_input_yaml,u0,input_yaml_base,integration) 

      elif model_type in ('LOpInf','LOpInf-SpML'):
        models = []
        for i in model_sample_indices(model_type,ml_input_yaml):
          checkpoint = input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_sample_' + str(i) + '.pt'
          models.append(torch.load(checkpoint,weights_only=False))
        u_snapshots,t_snapshots = solve_lagrangian_opinf_rom(
            Phi,models,params,fom_input_yaml,u0
        )

      else: 
        model = torch.load(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_model.pt')

        with open(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_feature_normalizer.pickle', 'rb') as file:
          feature_normalizer = pickle.load(file)
        with open(input_yaml_base['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_response_normalizer.pickle', 'rb') as file:
          response_normalizer = pickle.load(file)

        u_snapshots,t_snapshots = solve_ml_rom(Phi,model,params,feature_normalizer,response_normalizer,mu_mean,mu_std,fom_input_yaml,u0,integration) 
    uhat_snapshots = np.einsum('ij,i...->j...',Phi,u_snapshots) 
    return uhat_snapshots,Phi

def solve_galerkin_rom(Phi,myFom,params,fom_input_yaml,u0,integration='rk4'):
  N = Phi.shape[0]
  K = Phi.shape[1]
  if u0 is None:
    uhat = np.zeros(K)
  elif callable(u0):
    u0 = u0(params)
    uhat = Phi.transpose() @ u0
  else:
    uhat = Phi.transpose() @ u0 
  t = 0
  et = fom_input_yaml['end-time']
  u_snapshots = np.zeros((N,0))
  snapshot_collect_frequency = fom_input_yaml['snapshot-collect-frequency']
  t_snapshots = np.zeros(0)
  counter = 0
  dt = fom_input_yaml['dt']
  def reduced_rhs(state):
    return Phi.transpose() @ myFom.velocity(Phi @ state,params)
  while t <= et - dt/2:
    if (counter % snapshot_collect_frequency == 0):
      u_snapshots = np.append(u_snapshots,(Phi @ uhat)[:,None],axis=1)
      fhat = Phi.transpose() @ myFom.velocity(Phi @ uhat,params)
      t_snapshots = np.append(t_snapshots,t)

    
    uhat = first_order_step(reduced_rhs,uhat,dt,integration=integration)
    t += dt
    counter += 1
  return u_snapshots,t_snapshots

def solve_opinf_rom(Phi,model,params,fom_input_yaml,u0,parametric,integration=None):
    t = 0.
    et = float(fom_input_yaml['end-time'])
    snapshot_collect_frequency = int(fom_input_yaml['snapshot-collect-frequency']) 
    dt = float(fom_input_yaml['dt']) 
    nx = Phi.shape[0]
    K = Phi.shape[1]
    if integration is None:
      integration = fom_input_yaml.get('integration','rk4')
    #if u0 is None:
    #  uhat = np.zeros(K)
    #else:
    #  uhat = Phi.transpose() @ u0 
    if u0 is None:
      uhat = np.zeros(K)
    elif callable(u0):
      u0 = u0(params)
      uhat = Phi.transpose() @ u0
    else:
      uhat = Phi.transpose() @ u0 

    u_snapshots = np.zeros((nx,0))
    counter = 0
    t_snapshots = np.zeros(0)
    while t <= et - dt/2:
      if (counter % snapshot_collect_frequency == 0):
        u_snapshots = np.append(u_snapshots,(Phi @ uhat)[:,None],axis=1)
        t_snapshots = np.append(t_snapshots,t)
  
      if parametric:
        rhs = lambda state: model.rhs(0.,params,state).flatten()
        jacobian = lambda state: model.jacobian(0.,params,state)
      else:
        rhs = lambda state: model.rhs(0.,state).flatten()
        jacobian = lambda state: model.jacobian(0.,state)
      uhat = first_order_step(
          rhs,uhat,dt,integration=integration,jacobian=jacobian
      )
      t += dt
      counter += 1


    return u_snapshots,t_snapshots


def solve_nnopinf_rom(Phi,models,params,fom_input_yaml,u0,input_yaml_base,integration=None):
  for model in models:
    model.eval()
  ta = time.time()
  with torch.no_grad():
    t = 0.
    parameterized = input_yaml_base['machine-learning']['parameterized'] 
    et = float(fom_input_yaml['end-time'])
    snapshot_collect_frequency = int(fom_input_yaml['snapshot-collect-frequency']) 
    dt = float(fom_input_yaml['dt']) 
    nx = Phi.shape[0]
    K = Phi.shape[1]
    #if u0 is None:
    #  uhat = torch.tensor(np.zeros(K),dtype=torch.float64)
    #else:
    #  uhat = torch.tensor(Phi.transpose() @ u0,dtype=torch.float64)
    if u0 is None:
      uhat = np.zeros(K)
    elif callable(u0):
      u0 = u0(params)
      uhat = Phi.transpose() @ u0
    else:
      uhat = Phi.transpose() @ u0 
    uhat = torch.tensor(uhat,dtype=torch.float64)
    u_snapshots = np.zeros((nx,0))
    counter = 0
    t_snapshots = np.zeros(0)
    #parameters = torch.tensor( (params - params_mean)/(params_std + 1e-2) ,dtype=torch.float64)
    parameters = torch.tensor(params)[None]
    null_vector = torch.tensor(np.zeros((1,0)))
    if integration is None:
      integration = fom_input_yaml.get('integration','rk4')

    def reduced_rhs(state):
      input_v = torch.tensor(state,dtype=torch.float64)
      inputs = {'x':input_v[None],'mu':parameters,'null':null_vector}
      value = 0.0
      for model in models:
        value += model(inputs)[0].detach().numpy()
      return value / float(len(models))

    while t <= et - dt/2:
      if (counter % snapshot_collect_frequency == 0):
        u_snapshots = np.append(u_snapshots,(Phi @ uhat.detach().numpy())[:,None],axis=1)
        t_snapshots = np.append(t_snapshots,t)
 
      try:
        next_state = first_order_step(
            reduced_rhs,uhat.detach().numpy(),dt,integration=integration
        )
        if not np.all(np.isfinite(next_state)):
          raise FloatingPointError('time integration produced a non-finite state')
        uhat = torch.tensor(next_state,dtype=torch.float64)
      except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        num_steps = max(0,int(np.floor((et - dt/2)/dt + 1e-12)) + 1)
        snapshot_indices = np.arange(0,num_steps,snapshot_collect_frequency)
        print('WARNING: NNOpInf simulation failed at parameter '
              + np.array2string(np.asarray(params)) + '; '
              + f'{type(exc).__name__}: {exc}. '
              + 'Substituting an egregious finite solution.',flush=True)
        return (
            np.full(
                (nx,snapshot_indices.size),_EGREGIOUS_ROM_VALUE,dtype=float
            ),
            snapshot_indices * dt,
        )

      t += dt
      counter += 1
  return u_snapshots,t_snapshots


def solve_lagrangian_opinf_rom(Phi,models,params,fom_input_yaml,u0):
  """Integrate a LOpInf ensemble with implicit Newmark."""
  for model in models:
    model.eval()
  K = Phi.shape[1]
  if u0 is None:
    full_state = np.zeros(Phi.shape[0])
    full_velocity = np.zeros(Phi.shape[0])
  else:
    initial = u0(params) if callable(u0) else u0
    if isinstance(initial,(tuple,list)):
      if len(initial) != 2:
        raise ValueError("second-order initial conditions must be (q0, v0)")
      full_state,full_velocity = initial
    else:
      full_state = initial
      full_velocity = np.zeros_like(full_state)
  reduced_state = Phi.transpose() @ np.asarray(full_state)
  reduced_velocity = Phi.transpose() @ np.asarray(full_velocity)
  dt = float(fom_input_yaml['dt'])
  end_time = float(fom_input_yaml['end-time'])
  num_steps = int(np.floor(end_time/dt + 0.5))
  stepper = nnopinf.steppers.LagrangianNewmarkStepper(models)
  reduced_history = stepper.advance(
      reduced_state,reduced_velocity,dt,num_steps
  ).detach().numpy()
  collect = int(fom_input_yaml['snapshot-collect-frequency'])
  indices = np.arange(0,num_steps,collect,dtype=int)
  u_snapshots = Phi @ reduced_history[:,indices]
  t_snapshots = indices * dt
  return u_snapshots,t_snapshots


def solve_ml_rom(Phi,model,params,feature_normalizer,response_normalizer,params_mean,params_std,fom_input_yaml,u0,integration=None):
  model.eval()
  ta = time.time()
  with torch.no_grad():
    t = 0.
    et = float(fom_input_yaml['end-time'])
    snapshot_collect_frequency = int(fom_input_yaml['snapshot-collect-frequency']) 
    dt = float(fom_input_yaml['dt']) 
    nx = Phi.shape[0]
    K = Phi.shape[1]
    #if u0 is None:
    #  uhat = torch.tensor(np.zeros(K),dtype=torch.float64)
    #else:
    #  uhat = torch.tensor(Phi.transpose() @ u0,dtype=torch.float64)
    if u0 is None:
      uhat = np.zeros(K)
    elif callable(u0):
      u0 = u0(params)
      uhat = Phi.transpose() @ u0
    else:
      uhat = Phi.transpose() @ u0 
    uhat = torch.tensor(uhat,dtype=torch.float64)
    u_snapshots = np.zeros((nx,0))
    counter = 0
    t_snapshots = np.zeros(0)
    parameters = torch.tensor( (params - params_mean)/(params_std + 1e-2) ,dtype=torch.float64)
    if integration is None:
      integration = fom_input_yaml.get('integration','rk4')

    def reduced_rhs(state):
      features = torch.tensor(state,dtype=torch.float64)
      features = feature_normalizer.apply_scaling(features)
      features = torch.cat((features,parameters))
      value = model(features[None])[0]
      value = response_normalizer.apply_inverse_scaling(value)
      return value.detach().numpy()
    #parameters = torch.tensor( params ,dtype=torch.float64)

    while t <= et - dt/2:
      if (counter % snapshot_collect_frequency == 0):
        u_snapshots = np.append(u_snapshots,(Phi @ uhat.detach().numpy())[:,None],axis=1)
        t_snapshots = np.append(t_snapshots,t)
  
      uhat = torch.tensor(
          first_order_step(
              reduced_rhs,uhat.detach().numpy(),dt,integration=integration
          ),
          dtype=torch.float64,
      )
      t += dt
      counter += 1
  return u_snapshots,t_snapshots


def solve_entropy_ml_rom(rom,Phi,model,params,feature_normalizer,response_normalizer,params_mean,params_std,fom_input_yaml,u0,integration='rk4'):
  model.eval()
  if integration != 'rk4':
    raise ValueError("entropy only supports the 'rk4' integrator")
  ta = time.time()
  myFom = rom.myFom

  myFom.set_params(params)

  u0 = myFom.y0
  n = int(u0.size/3)
  u0 = np.reshape(u0,(3,n),order='F')
  u0 = conservative_to_entropy(u0).flatten('F')

  with torch.no_grad():
    t = 0.
    et = float(fom_input_yaml['end-time'])
    snapshot_collect_frequency = int(fom_input_yaml['snapshot-collect-frequency']) 
    dt = float(fom_input_yaml['dt']) 
    rk4const = np.array([1./4.,1./3.,1./2.,1.])
    nx = Phi.shape[0]
    n_vars = 3
    K = Phi.shape[1]
    PhiReshape = np.reshape(Phi,(n_vars,int(nx/n_vars),K),order='F')

    if u0 is None:
      uhat = torch.tensor(np.zeros(K),dtype=torch.float64)
    else:
      uhat = torch.tensor(Phi.transpose() @ u0,dtype=torch.float64)
    u_snapshots = np.zeros((nx,0))
    counter = 0
    t_snapshots = np.zeros(0)
    parameters = torch.tensor( (params - params_mean)/(params_std + 1e-2) ,dtype=torch.float64)
    #parameters = torch.tensor( params ,dtype=torch.float64)

    while t <= et:
      if (counter % snapshot_collect_frequency == 0):
        u_snapshots = np.append(u_snapshots,(Phi @ uhat.detach().numpy())[:,None],axis=1)
        t_snapshots = np.append(t_snapshots,t)
  
      uhat0 = uhat*1.
      for i in range(0,4):
        features = uhat*1.
        features = feature_normalizer.apply_scaling(features) 
        v = Phi @ uhat.detach().numpy()
        v = np.reshape(v,(3,int(v.size/3.)),'F')
        u = entropy_to_conservative(v*1.)
        mass = compute_dUdV(v)
        myFom.problem.rightHandSide(u.flatten('F'),0.,myFom.f)
        rhs = Phi.transpose() @ myFom.f 
        reducedMass = np.einsum('mni...,nik...->mik...',mass,PhiReshape)
        reducedMass = np.einsum('mil...,mik...->lk...',PhiReshape,reducedMass)

        #features = torch.cat((features,parameters))
        M,Kop = model.forward_operators(features[None])
        M = M[0] 
        M += 1e-2*np.eye(K)
        Kop = Kop[0] 
        # M [unp1 - un] = Ku
        #print('M',M)
        #print('M exact', reducedMass)
        rhs = np.linalg.solve(reducedMass,Kop @ uhat)
        uhat = uhat0 + dt*rk4const[i]*rhs
      t += dt
      counter += 1
  return u_snapshots,t_snapshots



def fom_driver(myFom,input_yaml_base):
  input_yaml = input_yaml_base['fom'] 
  print('Running FOM training')
  params = parameter_reader(input_yaml_base['parameter-training-file'])
  for i in range(0,params.shape[0]):
      param = params[i]
      usnaps,tsnaps,forcing,x,fsnaps = myFom.solve(param,input_yaml)
      if i == 0:
        usnaps_g = usnaps[:,:,None]
        fsnaps_g = fsnaps[:,:,None]
        params_g = (  param[:,None]*np.ones(tsnaps.size)[None,:] )[:,:,None]

      else:
        usnaps_g = np.append(usnaps_g,usnaps[:,:,None],axis=2)
        fsnaps_g = np.append(fsnaps_g,fsnaps[:,:,None],axis=2)
        params_tmp = param[:,None]*np.ones(tsnaps.size)[None,:]
        params_g = np.append(params_g,params_tmp[:,:,None],axis=2)

  print('FINISHED',usnaps_g.shape,usnaps.shape)
  saveDir = input_yaml_base['output-directory'] + '/' + input_yaml['fom-output-directory']
  makeRecursiveDirsIfNeeded(saveDir)
  np.savez(saveDir + '/fom_snapshots_training',u=usnaps_g,t=tsnaps,params=params_g,forcing=forcing,x=x,f=fsnaps_g)

  print('Running FOM testing')
  params = parameter_reader(input_yaml_base['parameter-testing-file'])
  print('HERE',params,params.shape)
  for i in range(0,params.shape[0]):
      param = params[i]
      usnaps,tsnaps,forcing,x,fsnaps = myFom.solve(param,input_yaml)
      if i == 0:
        usnaps_g = usnaps[:,:,None]
        fsnaps_g = fsnaps[:,:,None]
        params_g = (  param[:,None]*np.ones(tsnaps.size)[None,:] )[:,:,None]

      else:
        usnaps_g = np.append(usnaps_g,usnaps[:,:,None],axis=2)
        fsnaps_g = np.append(fsnaps_g,fsnaps[:,:,None],axis=2)
        params_tmp = param[:,None]*np.ones(tsnaps.size)[None,:]
        params_g = np.append(params_g,params_tmp[:,:,None],axis=2)

  saveDir = input_yaml_base['output-directory'] + '/' + input_yaml['fom-output-directory']
  makeRecursiveDirsIfNeeded(saveDir)
  np.savez(saveDir + '/fom_snapshots_testing',u=usnaps_g,t=tsnaps,params=params_g,forcing=forcing,x=x,f=fsnaps_g)



def _load_rom_basis(model_type,model_name,input_yaml_base):
  ml_input_yaml = input_yaml_base['machine-learning']
  training_directory = (input_yaml_base['output-directory'] + '/'
                        + ml_input_yaml['training-output-directory'] + '/')
  if model_type == 'ROM':
    basis_name = 'ROM' + model_name[3::] + '_basis.npz'
  else:
    basis_name = model_name + '_basis.npz'
  return np.load(training_directory + basis_name)['Phi']


def _solve_rom_safely(myRom,model_type,model_name,param,input_yaml_base,u0,
                      expected_snapshots,phase):
  """Return a finite ROM trajectory, substituting an egregious one on failure."""
  try:
    u_snapshots,Phi = myRom.solve(
        model_type,model_name,param,input_yaml_base,u0
    )
  except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
    Phi = _load_rom_basis(model_type,model_name,input_yaml_base)
    failure = f'{type(exc).__name__}: {exc}'
  else:
    valid_shape = (
        isinstance(u_snapshots,np.ndarray)
        and u_snapshots.ndim == 2
        and u_snapshots.shape == (Phi.shape[1],expected_snapshots)
    )
    if valid_shape and np.all(np.isfinite(u_snapshots)):
      return u_snapshots,Phi
    if not valid_shape:
      failure = ('invalid trajectory shape '
                 + str(getattr(u_snapshots,'shape',None)))
    else:
      failure = 'trajectory contains NaN or infinite values'

  print('WARNING: ' + phase + ' simulation failed for ' + model_name
        + ' at parameter ' + np.array2string(np.asarray(param)) + '; '
        + failure + '. Substituting an egregious finite solution.',flush=True)
  u_snapshots = np.full(
      (Phi.shape[1],expected_snapshots),_EGREGIOUS_ROM_VALUE,dtype=float
  )
  return u_snapshots,Phi


def rom_driver(myRom,input_yaml_base,u0=None):
  print('Simulating ROMs')
  input_yaml = input_yaml_base['roms']
  ml_input_yaml = input_yaml_base['machine-learning']
  input_yaml['full-training-output-directory'] = input_yaml_base['output-directory'] + '/' + input_yaml['rom-output-directory']
  KsToRun = np.array(input_yaml['reduced-basis-dimensions'],dtype='int')
  stop_times = np.array(input_yaml['training-stop-times'],dtype=float)
  model_types = input_yaml['model-types'] 
  ## Training
  params = parameter_reader(input_yaml_base['parameter-training-file'])
  fom_solution = np.load(input_yaml_base['output-directory'] + '/' + input_yaml_base['fom']['fom-output-directory'] + '/fom_snapshots_training.npz')

  stop_times_flatten,model_types_flatten,K_flatten = mesh_grid_flatten(stop_times,model_types,KsToRun)
  n_models = stop_times_flatten.size
  runs_per_rank = int(np.ceil(n_models*1./Nranks))
  starting_rank = min(runs_per_rank * rank,n_models)
  ending_rank = min( runs_per_rank*(rank+1),n_models)

  for model_no in range(starting_rank,ending_rank):
          stop_time = stop_times_flatten[model_no]
          model_type = model_types_flatten[model_no]
          K = K_flatten[model_no]
          model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) 
          for i in range(0,params.shape[0]):
            param = params[i]
            t0 = time.time()
            u_snapshots,Phi = _solve_rom_safely(
                myRom,model_type,model_name,param,input_yaml_base,u0,
                fom_solution['u'].shape[1],'training'
            )
            run_time = time.time() - t0
            if i == 0:
              u_snaps_g = u_snapshots[:,:,None]
            else:
              u_snaps_g = np.append(u_snaps_g,u_snapshots[:,:,None],axis=2)
          print('Passed')
#          error_space_time = np.linalg.norm(u_snaps_g - fom_solution['u']) / np.linalg.norm(fom_solution['u'])
#          error_last_step = np.linalg.norm(u_snaps_g[:,-1] - fom_solution['u'][:,-1]) /  np.linalg.norm(fom_solution['u'][:,-1])  
          error_space_time = np.linalg.norm(np.einsum('ij,j...->i...',Phi,u_snaps_g) - fom_solution['u']) / np.linalg.norm(fom_solution['u'])
          error_last_step = np.linalg.norm(np.einsum('ij,j...->i...',Phi,u_snaps_g)[:,-1] - fom_solution['u'][:,-1]) /  np.linalg.norm(fom_solution['u'][:,-1])  

          saveDir = input_yaml_base['output-directory'] + '/' + input_yaml['rom-output-directory'] + '/'
          makeRecursiveDirsIfNeeded(saveDir)
          np.savez(saveDir + model_name + '_training' ,u=u_snaps_g,error_space_time = error_space_time,error_last_step = error_last_step,run_time=run_time,Phi=Phi) 
  ## Testing
  params = parameter_reader(input_yaml_base['parameter-testing-file'])
  fom_solution = np.load(input_yaml_base['output-directory'] + '/' + input_yaml_base['fom']['fom-output-directory'] + '/fom_snapshots_testing.npz')
  for model_no in range(starting_rank,ending_rank):
          stop_time = stop_times_flatten[model_no]
          model_type = model_types_flatten[model_no]
          K = K_flatten[model_no]
          model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) 
          for i in range(0,params.shape[0]):
            param = params[i]
            t0 = time.time()
            u_snapshots,Phi = _solve_rom_safely(
                myRom,model_type,model_name,param,input_yaml_base,u0,
                fom_solution['u'].shape[1],'testing'
            )
            run_time = time.time() - t0
            if i == 0:
              u_snaps_g = u_snapshots[:,:,None]
            else:
              u_snaps_g = np.append(u_snaps_g,u_snapshots[:,:,None],axis=2)
          print('Passed')
          error_space_time = np.linalg.norm(np.einsum('ij,j...->i...',Phi,u_snaps_g) - fom_solution['u']) / np.linalg.norm(fom_solution['u'])
          error_last_step = np.linalg.norm(np.einsum('ij,j...->i...',Phi,u_snaps_g)[:,-1] - fom_solution['u'][:,-1]) /  np.linalg.norm(fom_solution['u'][:,-1])  
          saveDir = input_yaml_base['output-directory'] + '/' + input_yaml['rom-output-directory'] + '/'
          makeRecursiveDirsIfNeeded(saveDir)
          np.savez(saveDir + model_name + '_testing' ,u=u_snaps_g,error_space_time = error_space_time,error_last_step = error_last_step, run_time=run_time,Phi=Phi) 
