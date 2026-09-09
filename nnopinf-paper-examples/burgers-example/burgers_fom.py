import numpy as np
from matplotlib import pyplot as plt
import yaml
import argparse
import sys
sys.path.append('../src/')
from utilities import makeRecursiveDirsIfNeeded,parameter_reader
from drivers import fom_driver
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
  fL[:] = fsym[:] #- 0.5*(lamRoe + np.abs(du)/6.)*du
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
    self.forcing = np.zeros(self.x.size)
    self.dx = self.x[1] - self.x[0]
    self.u0 = 0.01*(np.sin(4.*self.x) + 2.*np.cos(6.*self.x)) + 2.5 

  def velocity(self,u,params):
    f = -upwind_deriv(u,self.dx) 
    return f


  def solve(self,params,input_yaml):
    u = self.u0 
    t = 0.
    nu = params[0]
    dt = float(input_yaml['dt']) 
    rk4const = np.array([1./4.,1./3.,1./2.,1.])
    u_snapshots = np.zeros((self.nx,0))
    f_snapshots = np.zeros((self.nx,0))
    counter = 0
    t_snapshots = np.zeros(0)
    snapshot_collect_frequency = int(input_yaml['snapshot-collect-frequency']) 
    while t <= float(input_yaml['end-time']) - dt/2.:
      if (counter % snapshot_collect_frequency == 0):
        u_snapshots = np.append(u_snapshots,u[:,None],axis=1)
        f_snapshots = np.append(f_snapshots,myFom.velocity(u,params)[:,None],axis=1)
        t_snapshots = np.append(t_snapshots,t)
  
      u0 = u*1.
      for i in range(0,4):
        f = myFom.velocity(u,params) 
        u = u0 + dt*rk4const[i]*f
      t += dt
      counter += 1
    return u_snapshots,t_snapshots,self.forcing,self.x,f_snapshots


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file",required=True)
  args = parser.parse_args()
  with open(args.i) as f:
        input_yaml_base = yaml.safe_load(f)

  myFom = burgers_fom(2.*np.pi,int(input_yaml_base['fom']['nx']))
  fom_driver(myFom,input_yaml_base) 
