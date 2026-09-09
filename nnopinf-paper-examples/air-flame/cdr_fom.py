import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import copy
axis_font = {'size':'20'}
import matplotlib.pyplot as plt 
axis_font = {'size':16,'family':'serif'}
import numpy as np
import yaml
import time
import argparse
import sys
sys.path.append('../src/')
from utilities import makeRecursiveDirsIfNeeded,parameter_reader
from drivers import fom_driver

def diff2(u,dx,dy):
  # compute d^2u/dx^2 + d^2u/dy^2
  result = np.zeros(u.shape,dtype=u.dtype)
  result[:,1:-1] += (u[:,2::] - 2.*u[:,1:-1] + u[:,0:-2]) / (dx**2)
  result[:,:,1:-1] += (u[:,:,2::] - 2.*u[:,:,1:-1] + u[:,:,0:-2]) / (dy**2)
  return result 

  

def diff(u,dx,dy):
  # compute d^2u/dx^2 + d^2u/dy^2
  result_x,result_y = np.zeros(u.shape,dtype=u.dtype),np.zeros(u.shape,dtype=u.dtype)
  result_x[:,1::]  += (u[:,1::] - u[:,0:-1]) / (dx)
  result_x[:,2::] = (3./2.*u[:,2::] - 2.*u[:,1:-1] + 0.5*u[:,0:-2])/dx
  return result_x,result_y


class advection_diffusion_fom_2d:
  def __init__(self,nx,ny):
    self.Lx = 1.8 #cm
    self.Ly = 0.9 #cm
    self.dx = self.Lx / nx
    self.dy = self.Ly / ny
    self.Nx = nx
    self.Ny = ny
    self.x = np.linspace(0,self.Lx,self.Nx)
    self.y = np.linspace(0,self.Ly,self.Ny)
    #self.x,self.y = np.meshgrid(self.x,self.y,indexing='ij')
    self.gamma_1_y_indices = self.y >= 6.e-1
    self.gamma_2_y_indices = np.logical_and(self.y < 6.e-1,self.y > 3.e-1)
    self.gamma_3_y_indices = self.y <= 3.e-1
    #self.Ad = BuildDiffusionMatrix(self.Nx,self.Ny,self.dx,self.dy) 

    ## Non-dimensionalization
    self.Tstar = 300.0

    self.kappa_ =  2.0# / 100**2
    self.beta_ = np.array([50.0,0.0]) #/ 100.0 #m/s
    self.W_ = np.array([2.016,31.9,18.]) # g/mol
    self.rho_ = 1.39e-3
    self.Q_ = 9800 / self.Tstar #K
    self.R_ = 8.314472 * self.Tstar
 
    self.nu_ = np.array([2,1,-2])


    ## Boundary conditions
    self.T_ambient = 300.0
    self.T_inlet = 950.0
    self.W_H2_inlet = 0.0282
    self.W_O2_inlet = 0.2259
    self.W_H2O_inlet = 0.0     

  
    self.u0 = np.zeros((4,self.Nx,self.Ny))
    self.u0[-1] = 300.0 / self.Tstar

    self.J_lin = None

  def apply_bcs(self,u):
    u[:,:,0] = 1.0*u[:,:,1]
    u[:,:,-1] = 1.0*u[:,:,-2]
    u[:,-1,:] = 1.0*u[:,-2,:]

    u[0,0,self.gamma_1_y_indices] = 0.
    u[1,0,self.gamma_1_y_indices] = 0.
    u[2,0,self.gamma_1_y_indices] = 0.
    u[3,0,self.gamma_1_y_indices] = self.T_ambient/self.Tstar 

    u[0,0,self.gamma_2_y_indices] = self.W_H2_inlet 
    u[1,0,self.gamma_2_y_indices] = self.W_O2_inlet
    u[2,0,self.gamma_2_y_indices] = self.W_H2O_inlet
    u[3,0,self.gamma_2_y_indices] = self.T_inlet / self.Tstar

    u[0,0,self.gamma_3_y_indices] = 0.
    u[1,0,self.gamma_3_y_indices] = 0.
    u[2,0,self.gamma_3_y_indices] = 0.
    u[3,0,self.gamma_3_y_indices] = self.T_ambient / self.Tstar
    return u

  def apply_bcs_to_residual(self,u,r):
    u = np.reshape(u,(4,self.Nx,self.Ny))
    r = np.reshape(r,(4,self.Nx,self.Ny))

    r[:,:,0] = u[:,:,1] - u[:,:,0]
    r[:,:,-1] = u[:,:,-1] - u[:,:,-2]
    r[:,-1,:] = u[:,-1,:] - u[:,-2,:]
    r[0,0,self.gamma_1_y_indices] = u[0,0,self.gamma_1_y_indices] - 0.
    r[1,0,self.gamma_1_y_indices] = u[1,0,self.gamma_1_y_indices] - 0.
    r[2,0,self.gamma_1_y_indices] = u[2,0,self.gamma_1_y_indices] - 0.
    r[3,0,self.gamma_1_y_indices] = u[3,0,self.gamma_1_y_indices] - self.T_ambient / self.Tstar

    r[0,0,self.gamma_2_y_indices] = u[0,0,self.gamma_2_y_indices] - 0.0282
    r[1,0,self.gamma_2_y_indices] = u[1,0,self.gamma_2_y_indices] - 0.2259
    r[2,0,self.gamma_2_y_indices] = u[2,0,self.gamma_2_y_indices] - 0.0
    r[3,0,self.gamma_2_y_indices] = u[3,0,self.gamma_2_y_indices] - self.T_inlet / self.Tstar

    r[0,0,self.gamma_3_y_indices] = u[0,0,self.gamma_3_y_indices] - 0.
    r[1,0,self.gamma_3_y_indices] = u[1,0,self.gamma_3_y_indices] - 0.
    r[2,0,self.gamma_3_y_indices] = u[2,0,self.gamma_3_y_indices] - 0.
    r[3,0,self.gamma_3_y_indices] = u[3,0,self.gamma_3_y_indices] - self.T_ambient / self.Tstar
    return r 


  def velocity_lin(self,u,params):
    u = np.reshape(u,(4,self.Nx,self.Ny))
    kappa = self.kappa_
    beta = self.beta_
   
    dux,duy = diff(u,self.dx,self.dy) 
    lhs = -kappa*diff2(u,self.dx,self.dy) + beta[0]*dux 
    return -lhs 

  def velocity(self,u,params):
    t0 = time.time()
    u = np.reshape(u,(4,self.Nx,self.Ny))
    rhs_lin = self.velocity_lin(u,params)
    A = params[0]*1e12
    E = params[1]*1e3
    reaction = np.zeros(u.shape,dtype=u.dtype)
    common = (self.rho_*u[0]/self.W_[0])**self.nu_[0]*(self.rho_*u[1]/self.W_[1])**self.nu_[1]*A*np.exp( - E /(self.R_ * u[-1] ))
    for i in range(0,3):
      reaction[i] = -self.nu_[i]*(self.W_[i]/self.rho_)*common
    reaction[-1] = reaction[-2] * self.Q_ 
    velocity = rhs_lin + reaction
    return velocity.flatten() 

  def compute_reaction_jacobian(self,u,params):
    """Compute the Jacobian of the reaction with respect to u."""
    nu = self.nu_
    u = np.reshape(u,(4,self.Nx,self.Ny))

    mask = np.ones((self.Nx,self.Ny))
    mask[0,:] = 0.
    mask[-1,:] = 0.
    mask[:,0] = 0.
    mask[:,-1] = 0.

    kappa = self.kappa_
    beta = self.beta_ 
    rho = self.rho_ 
   
    W = self.W_
    Q = self.Q_/self.Tstar
    A = params[0]*1e12
    E = params[1]*1e3
    R = self.R_ * self.Tstar


    data = np.zeros(0)
    rows_g = np.zeros(0)
    cols_g = np.zeros(0)
    nx = self.Nx
    ny = self.Ny
    for i in range(3):
        # Reaction term for R_i
        data = np.append(data,  -mask*nu[i] * (W[i] / rho) * (rho / W[0])**nu[1] * (rho * u[1] / W[1])**nu[1] * A * nu[0] * (rho / W[0]) * (u[0]**(nu[0]-1)) * np.exp(-E / (R * u[-1])) )
        rows_g = np.append(rows_g,np.arange(0*nx*ny,(0+1)*nx*ny))
        cols_g = np.append(cols_g,np.arange(i*nx*ny,(i+1)*nx*ny))
          

        data = np.append(data,-mask*nu[i] * (W[i] / rho) * (rho * u[0] / W[0])**nu[0] * A * nu[1] * (rho / W[1]) * (u[1]**(nu[1]-1)) * np.exp(-E / (R * u[-1])))
        rows_g = np.append(rows_g,np.arange(1*nx*ny,(1+1)*nx*ny))
        cols_g = np.append(cols_g,np.arange(i*nx*ny,(i+1)*nx*ny))


        data = np.append(data,  -mask*nu[i] * (W[i] / rho) * (rho * u[0] / W[0])**nu[0] * (rho * u[1] / W[1])**nu[1] * A * (E / (R * u[-1]**2)) * np.exp(-E / (R * u[-1]))) 
        rows_g = np.append(rows_g,np.arange(3*nx*ny,(3+1)*nx*ny))
        cols_g = np.append(cols_g,np.arange(i*nx*ny,(i+1)*nx*ny))

    for i in range(3,4):
        # Reaction term for R_i
        data = np.append(data,  -mask*Q*nu[-1] * (W[-1] / rho) * (rho / W[0])**nu[1] * (rho * u[1] / W[1])**nu[1] * A * nu[0] * (rho / W[0]) * (u[0]**(nu[0]-1)) * np.exp(-E / (R * u[-1])) )
        rows_g = np.append(rows_g,np.arange(0*nx*ny,(0+1)*nx*ny))
        cols_g = np.append(cols_g,np.arange(i*nx*ny,(i+1)*nx*ny))
          

        data = np.append(data,-mask*Q*nu[-1] * (W[-1] / rho) * (rho * u[0] / W[0])**nu[0] * A * nu[1] * (rho / W[1]) * (u[1]**(nu[1]-1)) * np.exp(-E / (R * u[-1])))
        rows_g = np.append(rows_g,np.arange(1*nx*ny,(1+1)*nx*ny))
        cols_g = np.append(cols_g,np.arange(i*nx*ny,(i+1)*nx*ny))

        data = np.append(data,  -mask*Q*nu[-1] * (W[-1] / rho) * (rho * u[0] / W[0])**nu[0] * (rho * u[1] / W[1])**nu[1] * A * (E / (R * u[-1]**2)) * np.exp(-E / (R * u[-1]))) 
        rows_g = np.append(rows_g,np.arange(3*nx*ny,(3+1)*nx*ny))
        cols_g = np.append(cols_g,np.arange(i*nx*ny,(i+1)*nx*ny))

    N = u.size
    j_sparse = scipy.sparse.csr_matrix((data,(cols_g,rows_g)),shape=(N,N))
    return j_sparse

  def residual_lin(self,unp1,params):
    scheme = 'Crank-Nicolson'
    unp1 = np.reshape(unp1,(4,self.Nx,self.Ny))
    fnp1 = self.velocity_lin(unp1,params)
    if scheme == 'Implicit-Euler':
      resid = unp1  - fnp1*self.dt
    elif scheme == 'Crank-Nicolson':
      resid = unp1  - 0.5*fnp1*self.dt
    else:
      print('Scheme not recognized')
      sys.exit()

    resid = self.apply_bcs_to_residual(unp1,resid)
    return resid.flatten()


  def jacobian_lin(self,unp1,params):
    n = unp1.size
    j = np.zeros((n,n))
    eps = 1.e-15
    v0 = self.residual_lin(unp1,params)
    for i in range(0,n):
      unp1[i] += 1j*eps
      vp = self.residual_lin(unp1,params)
      j[:,i] = np.imag(vp)/eps
      unp1[i] -= 1j*eps
    return j

  def residual(self,unp1,un,fn,params):
    scheme = 'Crank-Nicolson'
    unp1 = np.reshape(unp1,(4,self.Nx,self.Ny))
    fnp1 = self.velocity(unp1,params)
    #print(params)
    if scheme == 'Implicit-Euler':
      resid = (unp1.flatten() - un.flatten()) - fnp1.flatten()*self.dt
    elif scheme == 'Crank-Nicolson':
      resid = (unp1.flatten() - un.flatten()) - 0.5*(fnp1.flatten() + fn.flatten())*self.dt
    resid = self.apply_bcs_to_residual(unp1,resid)
    return resid.flatten()
  
  def jacobian(self,unp1,params):
    if self.J_lin is None:
      u_cs = np.zeros(unp1.size,dtype='complex')
      self.J_lin = scipy.sparse.csr_matrix(self.jacobian_lin(u_cs.flatten(),params))

    n = unp1.size
    j = np.zeros((n,n))
   
    j_reaction = self.compute_reaction_jacobian(unp1,params)
    j2 = -self.dt*j_reaction + self.J_lin
    return j2


  def solve(self,params,input_yaml):
    snapshot_collect_frequency = int(input_yaml['snapshot-collect-frequency'])

    # Explicit time stepping w/ zero ICs 
    u = self.u0*1.0 
    unp1 = u*1.
    u = self.apply_bcs(u) 
    rk4const = np.array([1./4.,1./3.,1./2.,1.])
  
    u_history = np.zeros((u.size,0))
    f_history = np.zeros((u.size,0))
  
    t = 0.
    t_history = []
    dum_vec = u.flatten()*0.
    counter = 0
    et = input_yaml['end-time']
    dt = input_yaml['dt'] 
    self.dt = dt

    scheme = 'Crank-Nicolson'
   
    while t <= et - dt/2.:
      if counter % snapshot_collect_frequency == 0:
        t_history.append(t)
        u_history = np.append(u_history,u.flatten()[:,None],axis=1 ) 
        f = self.velocity(u,params)
        f_history = np.append(f_history,f.flatten()[:,None],axis=1 ) 
  
      def my_newton(x,xn,fn,params):
        r = self.residual(x,xn,fn,params)
        r0_norm = np.linalg.norm(r)
        iteration = 0
        max_its = 80
        while np.linalg.norm(r)/r0_norm > 1e-6 and iteration < max_its:
          t0 = time.time()
          J = self.jacobian(x,params)
          t1 = time.time()
          dx = scipy.sparse.linalg.spsolve(J,-r)
          #print(f'Relative residual norm: {np.linalg.norm(r)/r0_norm:.6f}, dx: {np.linalg.norm(dx):.6f}, iteration: {iteration}') 
          t3 = time.time()
  
          x = x + dx
          r = self.residual(x,xn,fn,params)
          t4 = time.time()
          iteration += 1
        if iteration == max_its:
          x /= 0. # return nan
        return x[:]



      un = u*1.
      fn = self.velocity(u,params)
      u = my_newton(u.flatten(),un.flatten(),fn,params)
      u = np.reshape(u,un.shape)
      t += dt
      #print(t)
      counter += 1

    return u_history,np.array(t_history),dum_vec,dum_vec,f_history


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file",required=True)
  args = parser.parse_args()
  with open(args.i) as f:
        input_yaml_base = yaml.safe_load(f)

  nx = int(input_yaml_base['fom']['nx'])
  ny = int(input_yaml_base['fom']['ny'])
  myFom = advection_diffusion_fom_2d(nx,ny)
  #u = myFom.solve(np.array([2.5 ,5.85]),input_yaml_base['fom'])
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
