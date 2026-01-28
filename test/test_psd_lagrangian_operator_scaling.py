import os
import sys

import numpy as np
import torch

# Ensure local package import works when running pytest from repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nnopinf.variables import Variable
from nnopinf.operators import PsdLagrangianOperator


def test_psd_lagrangian_operator_scaling_matches_normalized():
    torch.manual_seed(0)
    np.random.seed(0)

    x_var = Variable(size=3, name="x")
    mu_var = Variable(size=2, name="mu")

    op = PsdLagrangianOperator(
        acts_on=x_var,
        depends_on=(x_var,),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
        activation=torch.tanh,
        positive=True,
    )

    batch = 7
    x_raw = torch.randn(batch, x_var.get_size(), dtype=torch.float64)
    mu_raw = torch.randn(batch, mu_var.get_size(), dtype=torch.float64)

    x_scaling = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    mu_scaling = torch.tensor([5.0, 6.0], dtype=torch.float64)
    y_scaling = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64)

    inputs_scaled = {
        "x": x_raw / x_scaling,
        "mu": mu_raw / mu_scaling,
    }
    print(inputs_scaled['x'].shape,inputs_scaled['mu'].shape)
    y_scaled, jac_scaled = op.forward(inputs_scaled, return_jacobian=True)
    op.set_scalings({"x": x_scaling, "mu": mu_scaling}, y_scaling)

    inputs_raw = {"x": x_raw, "mu": mu_raw}
    y_raw, jac_raw = op.forward(inputs_raw, return_jacobian=True)

    expected_y_raw = y_scaled * y_scaling
    assert torch.allclose(y_raw, expected_y_raw, rtol=1e-10, atol=1e-10)

    #row_scale = (y_scaling / x_scaling).unsqueeze(0).unsqueeze(2)
    #col_scale = (1.0 / x_scaling).unsqueeze(0).unsqueeze(1)
    #expected_jac_raw = jac_scaled * row_scale * col_scale
    #assert torch.allclose(jac_raw, expected_jac_raw, rtol=1e-10, atol=1e-10)
