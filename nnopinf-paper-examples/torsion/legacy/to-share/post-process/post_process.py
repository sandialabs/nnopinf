import numpy as np
import normaopinf
import matplotlib.pyplot as plt
axis_font = {'size':'20'}
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif"
})

class fom_solution:
  def __init__(self,solution_directory,base_name,skip_files=1,slice_end=25):
    runs = [2]
    self.solution = None
    for i in range(len(runs)):
      directory = solution_directory 
      solution,t = normaopinf.readers.load_displacement_csv_files(directory,base_name,skip_files=skip_files)
      solution = solution[...,0:slice_end]
      if self.solution is None:
        self.solution = solution[...,None]
      else:
        self.solution = np.append(self.solution,solution[...,None],axis=2) 

class rom_solution:
  def __init__(self,solution_directory,base_name,skip_files=1,fom_solution=None,slice_end=25):
    runs = [0]
    self.solution = None
    for i in range(len(runs)):
      directory = solution_directory + '/'
      print(directory)
      solution,t = normaopinf.readers.load_displacement_csv_files(directory,base_name,skip_files=skip_files)
      solution = solution[...,0:slice_end]
      if self.solution is None:
        self.solution = solution[...,None]
      else:
        self.solution = np.append(self.solution,solution[...,None],axis=2) 

    if fom_solution is None:
      pass
    else:
      self.e = np.linalg.norm( self.solution - fom_solution.solution)/ ( np.linalg.norm(fom_solution.solution) + 1.e-4)

rom_dims = [8,12,16,20,40]
model_runs = [
  ("P-OpInf-A", "../linear/linear-opinf-operator-"),
  ("P-OpInf-AH", "../quadratic/quadratic-opinf-operator-"),
  ("NN-OpInf-NN", "../nn-forward/nn-operator-"),
  ("NN-OpInf-SPSD-Potential", "../psd-lagrangian/ortho-lr5em4-lagrangian-depth2-20k"),
#  ("galerkin", "../galerkin/galerkin_"),
]

plot_colors = {
  "P-OpInf-A": "green",
  "P-OpInf-AH": "purple",
  "NN-OpInf-SPSD-Potential": "red",
  "NN-OpInf-NN": "blue",
}

def compute_errors_by_model(slice_end):
  fom = fom_solution('../fom/', "torsion-in",skip_files=10,slice_end=slice_end)
  errors = {}
  for label, base_dir in model_runs:
    rom_sols = []
    for rom_dim in rom_dims:
      path = base_dir + str(rom_dim)
      if label == "NN-OpInf-SPSD-Potential" or label == 'NN-OpInf-NN':
        skip_files = 10
      else:
        skip_files = 1
      rom_sols.append(
        rom_solution(
          path + '/',
          "torsion-in",
          skip_files=skip_files,
          fom_solution=fom,
          slice_end=slice_end,
        )
      )
    errors[label] = [rom_sol.e for rom_sol in rom_sols]
    print(label)
    for rom_sol in rom_sols:
      print(rom_sol.e)
  return errors

errors_by_model = compute_errors_by_model(slice_end=25)
plt.figure()
markers = ['o','s','v','^','*','P']
i = 0
for label, errors in errors_by_model.items():
  plt.plot(rom_dims, errors, marker=markers[i], label=label, color=plot_colors.get(label), linewidth=2, markersize=11)
  i += 1
plt.xlabel(r'$\mathrm{Basis\ dimension}$',**axis_font)
plt.ylabel(r'$\mathrm{Relative\ error}$',**axis_font)
plt.ylim([5.e-3,2.0])
plt.grid()
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('convergence.pdf')
plt.show()

errors_by_model_future = compute_errors_by_model(slice_end=50)
plt.figure()
i = 0
for label, errors in errors_by_model_future.items():
  plt.plot(rom_dims, errors, marker=markers[i], label=label, color=plot_colors.get(label), linewidth=2, markersize=11)
  i += 1
plt.xlabel(r'$\mathrm{Basis\ dimension}$',**axis_font)
plt.ylabel(r'$\mathrm{Relative\ error}$',**axis_font)
plt.legend()
plt.ylim([5e-3,2.0])
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.savefig('convergence-future-state.pdf')
plt.show()
