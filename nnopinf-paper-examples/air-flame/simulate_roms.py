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
from cdr_fom import advection_diffusion_fom_2d
import os
import opinf
axis_font = {'size':'20'}


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file",required=True)
  args = parser.parse_args()
  with open(args.i) as f:
        input_yaml_base = yaml.safe_load(f)
  myFom = advection_diffusion_fom_2d(int(input_yaml_base['fom']['nx']),int(input_yaml_base['fom']['ny']))
  myRom = generic_rom(myFom)
  u0 = myFom.apply_bcs(myFom.u0)
  rom_driver(myRom,input_yaml_base,u0.flatten())
