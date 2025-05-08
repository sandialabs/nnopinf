import numpy as np
from matplotlib import pyplot as plt
import yaml
import argparse
import sys
import nnopinf
import nnopinf.operators as operators
import nnopinf.models as models
import nnopinf.training
import torch
from matplotlib import pyplot as plt

axis_font = {'size':'20'}
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif"
})


def upwind_deriv(u,dx):
  r = np.zeros(u.size)
  fsym = np.zeros(u.size)
  fsym[0] = 1./6.*(u[0]**2 + u[0]*u[-1] + u[-1]**2)
  fsym[1::] = 1./6.*(u[1::]**2 + u[1::]*u[0:-1] + u[0:-1]**2)

  lamRoe = np.zeros(u.size)
  du = np.zeros(u.size)

  lamRoe[1::] = 0.5*np.abs(u[1::] + u[0:-1])
  lamRoe[0] = 0.5*np.abs(u[0] + u[-1])

  du[1::] = (u[1::] - u[0:-1])
  du[0] = (u[0] - u[-1])

  fL = np.zeros(u.size)
  fL[:] = fsym[:] - 0.5*(lamRoe + np.abs(du)/6.)*du
  fR = np.roll(fL,-1)
  r = fR - fL
  return r/dx

class burgers_fom:
  def __init__(self,L,nx):
    self.L = L
    self.N = nx 
    self.nx = nx
    self.dx = self.L / self.N

    self.x = np.linspace(0,self.L - self.dx,self.nx)
    self.u0 = np.sin(self.x)

  def velocity(self,u):
    f = -upwind_deriv(u,self.dx)
    return f


  def solve(self,dt,end_time):
    u = self.u0 
    t = 0.
    rk4const = np.array([1./4.,1./3.,1./2.,1.])
    u_snapshots = np.zeros((self.nx,0))
    f_snapshots = np.zeros((self.nx,0))
    counter = 0
    t_snapshots = np.zeros(0)
    while t <= end_time - dt/2.:
      u_snapshots = np.append(u_snapshots,u[:,None],axis=1)
      t_snapshots = np.append(t_snapshots,t)
  
      u0 = u*1.
      for i in range(0,4):
        f = self.velocity(u) 
        u = u0 + dt*rk4const[i]*f
      t += dt
      counter += 1
    return u_snapshots,t_snapshots


class nn_opinf_rom:
  def __init__(self,model):
    self.model_ = model
    self.inputs_ = {}
  def velocity(self,u):
    self.inputs_['x'] = torch.tensor(u[None])
   
    ft = self.model_.forward(self.inputs_)[0].detach().numpy()  
    return ft


  def solve(self,u0,dt,end_time):
    u = u0 
    t = 0.
    rk4const = np.array([1./4.,1./3.,1./2.,1.])
    u_snapshots = np.zeros((u0.size,0))
    f_snapshots = np.zeros((u0.size,0))
    counter = 0
    t_snapshots = np.zeros(0)
    while t <= end_time - dt/2.:
      u_snapshots = np.append(u_snapshots,u[:,None],axis=1)
      t_snapshots = np.append(t_snapshots,t)
  
      u0 = u*1.
      for i in range(0,4):
        f = self.velocity(u) 
        u = u0 + dt*rk4const[i]*f
      t += dt
      counter += 1
    return u_snapshots,t_snapshots



if __name__=='__main__':

    ## Setup and solve the FOM
    nx = 500
    L = 2.*np.pi
    myFom = burgers_fom(L,nx)
    dt = 0.01
    et = 5.
    u,t = myFom.solve(dt,et)
  
    # Make POD basis
    Phi,s,_ = np.linalg.svd(u,full_matrices=False)
    relative_energy = np.cumsum(s**2) / np.sum(s**2)
    K = np.argmin(np.abs(relative_energy - 0.99999999))
    Phi = Phi[:,0:K]
  
    #Now do nnopinf
    uhat = Phi.transpose() @ u
    uhat_dot = (uhat[:,2::] - uhat[:,0:-2]) / (2.*dt)
    # Trim to make the same size
    uhat = uhat[...,1:-1]
    
    # Design operators
    n_hidden_layers = 1 
    n_neurons_per_layer = K
    n_inputs = K
    n_outputs = K

    # Design operators for the state. 
    # Create to be energy conserving , s.t. uhat_dot \le 0
    NpdMlp = operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SkewMlp = operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    NdMlp = operators.CompositeOperator([NpdMlp,SkewMlp])

    # Wrap operator for use by opinf model
    nd_operator =  models.WrappedOperatorForModel(operator=NdMlp,inputs=("x",),name="nd")
    my_operators = [nd_operator]
    my_model = models.OpInfModel( my_operators )

    # Train
    training_settings = nnopinf.training.get_default_settings()
    training_settings['batch-size'] = 500
    training_settings['num-epochs'] = 5000
    print("Settings are: ",training_settings)
    inputs = {}
    inputs['x'] = uhat.transpose()
    training_settings['x-normalization-strategy'] = 'MaxAbs'
    nnopinf.training.train(my_model,input_dict=inputs,y=uhat_dot.transpose(),training_settings=training_settings)
  
 
    # Do a forward pass of the model
    u0 = Phi.transpose() @ myFom.u0
    my_rom = nn_opinf_rom(my_model)
    urom,trom = my_rom.solve(u0,dt,et) 

    urom = Phi @ urom

    # Check errors and plot
    relative_error = np.linalg.norm(urom - u)/np.linalg.norm(u)
    print('Relative error = ' + str(relative_error))

    plt.close("all") 
    plt.plot(myFom.x,u[:,-1],color='black',label='FOM')
    plt.plot(myFom.x,urom[:,-1],'--',color='blue',label='NNOPINF-PD')
    plt.xlabel(r'$x$',**axis_font)
    plt.ylabel(r'$u(x)$',**axis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig('burgers_solution.pdf')
  

  
