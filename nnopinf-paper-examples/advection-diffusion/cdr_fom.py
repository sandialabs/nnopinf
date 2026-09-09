import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import copy
axis_font = {'size':'20'}
import matplotlib.pyplot as plt 
axis_font = {'size':16,'family':'serif'}
import numpy as np
import yaml
import argparse
import sys
sys.path.append('../src/')
from utilities import makeRecursiveDirsIfNeeded,parameter_reader
from drivers import fom_driver


class MySparseMatrixBuilder:
  '''
  This class contains information for iteratively constructing a 
  sparse matrix
  '''
  def __init__(self,N,M):
    # constructor for an NxM matrix 
    self.N = N
    self.M = M

  #create containers for non-zero row indices,
  #column indices, and their values
  row_indices = np.zeros(0,dtype='int')
  col_indices = np.zeros(0,dtype='int')
  values = np.zeros(0)

  #Function to add a new entry to the matrix
  def addEntry(self,row_index,col_index,value):
    self.row_indices = np.append(self.row_indices,row_index)
    self.col_indices = np.append(self.col_indices,col_index)
    self.values = np.append(self.values,value)

  #Assemble the sparse matrix
  def assemble(self):
    SparseMatrix = scipy.sparse.csr_matrix((self.values,(self.row_indices,self.col_indices)),(self.N,self.M))
    return SparseMatrix

def BuildAdvectionMatrices(Nx,Ny,dx,dy):
  '''
  This function builds in the system matrix for 
  advection via a second order upwind (backward) difference

  For 2D, we run through x points left to right,
  and then y points up and down
  '''
  Ax_builder = MySparseMatrixBuilder(Nx*Ny,Nx*Ny)
  Ay_builder = MySparseMatrixBuilder(Nx*Ny,Nx*Ny)

  #first compute du/dx and du/dy on interior 
  for j in range(1,Ny):
    for i in range(1,Nx):
      indx = i + j*Nx
      indx_im1 = indx - 1
      indx_im2 = indx - 2
      indx_jm1 = indx - Nx
      indx_jm2 = indx - Nx*2

      Ax_builder.addEntry(indx,indx,3./2./dx)
      Ax_builder.addEntry(indx,indx_im1,-2./dx)
      if (i > 1):
        Ax_builder.addEntry(indx,indx_im2,0.5/dx) #if i=1, the value is zero from the BCs
      
      Ay_builder.addEntry(indx,indx,3./2./dy)
      Ay_builder.addEntry(indx,indx_jm1,-2./dy)
      if (j > 1):
        Ay_builder.addEntry(indx,indx_jm2,0.5/dy)

  # now compute interior x at first y
  for j in range(0,1):
    for i in range(1,Nx):
      indx = i + j*Nx
      indx_im1 = indx - 1
      indx_im2 = indx - 2
      indx_jm1 = indx - Nx
      indx_jm2 = indx - Nx*2

      Ax_builder.addEntry(indx,indx,3./2./dx)
      Ax_builder.addEntry(indx,indx_im1,-2./dx)
      if (i > 1):
        Ax_builder.addEntry(indx,indx_im2,0.5/dx) #if i=1, the value is zero from the BCs

      Ay_builder.addEntry(indx,indx,1./dy)

  # now compute interior y at first x
  for j in range(1,Ny):
    for i in range(0,1):
      indx = i + j*Nx
      indx_im1 = indx - 1
      indx_im2 = indx - 2
      indx_jm1 = indx - Nx
      indx_jm2 = indx - Nx*2

      Ax_builder.addEntry(indx,indx,1./dx)

      Ay_builder.addEntry(indx,indx,3./2./dy)
      Ay_builder.addEntry(indx,indx_jm1,-2./dy)
      if (j > 1):
        Ay_builder.addEntry(indx,indx_jm2,0.5/dy)

  # now compute first x and y 
  Ax_builder.addEntry(0,0,1./dx)
  Ay_builder.addEntry(0,0,1./dy)

  Ax = Ax_builder.assemble()
  Ay = Ay_builder.assemble()

  return Ax,Ay

def BuildDiffusionMatrix(Nx,Ny,dx,dy):
  '''
  This function builds in the system matrix for 
  diffusion via a second order central difference
  '''
  A_builder = MySparseMatrixBuilder(Nx*Ny,Nx*Ny)
  #first compute du/dx and du/dy on interior 
  for j in range(0,Ny):
    for i in range(0,Nx):
      indx = i + j*Nx
      indx_im1 = indx - 1
      indx_ip1 = indx + 1
      indx_jm1 = indx - Nx
      indx_jp1 = indx + Nx

      A_builder.addEntry(indx,indx,-2./dx**2 - 2./dy**2)
      if (i > 0):
        A_builder.addEntry(indx,indx_im1,1./dx**2)
      if (i < Nx-1):
        A_builder.addEntry(indx,indx_ip1,1./dx**2)
      if (j > 0):
        A_builder.addEntry(indx,indx_jm1,1./dx**2)
      if (j < Ny-1):
        A_builder.addEntry(indx,indx_jp1,1./dx**2)
  A = A_builder.assemble()
  return A


class AdvectionDiffusionSystem:
  '''
  This class contains information for an advection diffusion system
  '''
  def __init__(self,Nx,Ny):
    self.Lx = 1.
    self.Ly = 1.
    self.N  = Nx*Ny
    self.Nx =  Nx
    self.Ny =  Ny
    self.dx = float(self.Lx) / float(self.Nx + 1)
    self.dy = float(self.Ly) / float(self.Ny + 1)
    self.g = np.ones(self.N)
    self.x = np.linspace(self.dx,self.Lx-self.dx,self.Nx)
    self.y = np.linspace(self.dy,self.Ly-self.dy,self.Ny)
    x,y = np.meshgrid(self.x,self.y)
    r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2 )
    self.g = 5.*np.exp(-4*r.flatten())
    self.A_diffusion = BuildDiffusionMatrix(self.Nx,self.Ny,self.dx,self.dy) 
    self.A_advection_x,self.A_advection_y = BuildAdvectionMatrices(self.Nx,self.Ny,self.dx,self.dy) 
    self.I = scipy.sparse.csr_matrix( np.eye(Nx*Ny) ) 


def residual(system,u,b,nu,sigma,eta):
  '''
  Given an AdvectionDiffusionSystem class and a solution,
  this function computes the residual
  '''
  A_nl = -(b[0]*system.A_advection_x.dot(u))*u - b[1]*(system.A_advection_y.dot(u))*u - sigma*u*np.exp(-eta*u**2)
  A_lin = system.A_diffusion*nu #- b[0]*system.A_advection_x - b[1]*system.A_advection_y 
  return A_lin.dot(u) + system.g + A_nl# - sigma*u*np.exp(-eta*u**2)  

def solveFom(system,b,nu,sigma,eta,dt,et,snapshot_collect_frequency):
  '''
  Given an AdvectionDiffusionSystem class, this function 
  solves the steady linear advection diffusion equation
  '''
  # Explicit time stepping w/ zero ICs 
  u = np.zeros(system.Nx*system.Ny)
  unp1 = u*1.

  rk4const = np.array([1./4.,1./3.,1./2.,1.])

  u_history = np.zeros((u.size,0))
  f_history = np.zeros((u.size,0))

  t = 0.
  t_history = []
  dum_vec = u*0.
  counter = 0
  while t <= et - dt/2.:
    if counter % snapshot_collect_frequency == 0:
      t_history.append(t)
      u_history = np.append(u_history,u[:,None],axis=1 ) 
      f = residual(system,u,b,nu,sigma,eta)
      f_history = np.append(f_history,f[:,None],axis=1 ) 
    u0 = u*1.
    for i in range(0,4):
      r = residual(system,u,b,nu,sigma,eta)
      u = u0 + dt*rk4const[i]*r
    t += dt
    counter += 1

  #LHS = system.A_diffusion*nu - b[0]*system.A_advection_x - b[1]*system.A_advection_y - sigma*system.I
  #RHS = -system.g
  print(np.shape(u_history),np.shape(f_history))
  return u_history,np.array(t_history),dum_vec,dum_vec,f_history

def computeGradientTangent(system,u,v):
  '''
  Given an AdvectionDiffusionSystem class, steady primal, and steady tangent solutions
  this function computes the gradient of the objective with respect to nu 
  '''
  return np.dot(system.C,v)

def computeGradientAdjoint(system,u,phi):
  '''
  Given an AdjointAdvectionDiffusionSystem class, steady primal, and steady adjoint solutions
  this function computes the gradient of the objective with respect to nu 
  '''
  dF = system.A_diffusion.dot(u) # dF/dnu
  return np.dot(dF,phi) 



class advection_diffusion_fom_2d:
  def __init__(self,nx,ny):
    self.AdvectionDiffusionSystem = AdvectionDiffusionSystem(nx,ny)

  def velocity(self,u,params):
    sigma = params[0]
    nu = params[1]
    theta = params[2]
    b = np.zeros(2)
    eta = params[3]
    bmag = 0.5
    angle = theta 
    b[0] = bmag*np.cos(angle)
    b[1] = bmag*np.sin(angle)
    r = residual(self.AdvectionDiffusionSystem,u,b,nu,sigma,eta)
    return r

  def solve(self,params,input_yaml):
    sigma = params[0]
    nu = params[1]
    theta = params[2]
    eta = params[3]

    b = np.zeros(2)
    dt = float(input_yaml['dt']) 
    et = float(input_yaml['end-time']) 
    bmag = 0.5
    angle = theta 
    b[0] = bmag*np.cos(angle)
    b[1] = bmag*np.sin(angle)
    snapshot_collect_frequency = int(input_yaml['snapshot-collect-frequency'])
    u_snapshots,t_snapshots,dum_vec,dum_vec,f_snapshots = solveFom(self.AdvectionDiffusionSystem,b,nu,sigma,eta,dt,et,snapshot_collect_frequency)
    return u_snapshots,t_snapshots,dum_vec,dum_vec,f_snapshots


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file",required=True)
  args = parser.parse_args()
  with open(args.i) as f:
        input_yaml_base = yaml.safe_load(f)

  nx = int(input_yaml_base['fom']['nx'])
  ny = int(input_yaml_base['fom']['ny'])
  myFom = advection_diffusion_fom_2d(nx,ny)
  fom_driver(myFom,input_yaml_base) 

'''
if __name__ == '__main__':
  # Main driver script
  Nx = 31
  Ny = Nx
  system = AdvectionDiffusionSystem(Nx,Ny)
  
  b = np.zeros(2)
  bmag = 0.5
  angle = np.pi/3.
  b[0] = bmag*np.cos(angle)
  b[1] = bmag*np.sin(angle)
  nu    = 1e-3
  sigma = 1.0
  
  # solve CDR equation
  et = 5.
  dt = 0.01
  u_history = solveFom(system,b,nu,sigma,dt,et,1) 

#  #TODO pad with zeros prior to postprocessing
#  u_full = np.zeros((Nx+2,Ny+2))
#  u_full[1:-1,1:-1] = u.reshape(Nx,Ny)
#  
#  Lx = system.Lx
#  Ly = system.Ly
#  
#  x = np.linspace(0.0,Lx,Nx+2)
#  y = np.linspace(0.0,Ly,Ny+2)
'''
