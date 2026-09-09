import numpy as np
import argparse
import yaml
import sys
sys.path.append('../src/')
from utilities import *
from matplotlib import pyplot as plt
import torch
from drivers import rom_driver, generic_rom
torch.set_default_dtype(torch.float64)
import os.path
from scipy.sparse.linalg import LinearOperator
import os
import opinf
from fd_nl_heat import *#heat_fom_fd2d
axis_font = {'size':'20'}


if __name__ == '__main__':

  import argparse
  import sys
  import yaml

  sys.path.append("../src/")
  from drivers import fom_driver

  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file", required=True)
  args = parser.parse_args()

  with open(args.i) as f:
      input_yaml_base = yaml.safe_load(f)

  fom_cfg = input_yaml_base.get("fom", {})
  nx = int(fom_cfg.get("nx", 40))
  ny = int(fom_cfg.get("ny", nx))
  dt = float(input_yaml_base.get("dt", 0.005))
  t_end = float(input_yaml_base.get("end-time", 0.2))
  gamma = float(fom_cfg.get("gamma", 50.0))

  model = heat_fom_fd2d(
        nx,
        ny,
        dt=dt,
        t_end=t_end,
        k_func=kappa,
        g_func=g_dirichlet,
        f_func=source_f,
        u0_func=initial_u,
        gamma=gamma,
  )
  myRom = generic_rom(model)
  rom_driver(myRom,input_yaml_base,model.u0)
