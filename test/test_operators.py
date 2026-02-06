import nnopinf
import nnopinf.operators
import torch
import numpy as np
from nnopinf.variables import Variable
from nnopinf.operators import PsdLagrangianOperator

DTYPE = torch.float64


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_scalings(x_dim, mu_dim, out_dim):
    x_scaling = torch.linspace(1.5, 2.5, x_dim, dtype=DTYPE)
    mu_scaling = torch.linspace(0.7, 1.3, mu_dim, dtype=DTYPE)
    out_scaling = torch.linspace(1.1, 1.9, out_dim, dtype=DTYPE)
    return x_scaling, mu_scaling, out_scaling


def _inputs_scaled(x_raw, mu_raw, x_scaling, mu_scaling):
    return {"x": x_raw / x_scaling, "mu": mu_raw / mu_scaling}


# -----------------
# SpdOperator tests
# -----------------

def test_spd_operator():
    _set_seed(1)

    x_input = nnopinf.Variable(size=3,name="x")
    mu_input = nnopinf.Variable(size=2,name="mu")
    NpdMlp = nnopinf.operators.SpdOperator(acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2,positive=False)

    _set_seed(1)

    SpdMlp = nnopinf.operators.SpdOperator(acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2,positive=True)

    x = torch.randn([5,3])
    mu = torch.randn([5,2])
    inputs = {}
    inputs['x'] = x
    inputs['mu'] = mu

    output,output_mat = SpdMlp.forward(inputs,return_jacobian=True)
    # Ensure y = Ax
    assert torch.allclose(output,torch.einsum('nij,nj->ni',output_mat , x))
    # Ensure SPD
    output_mat_i = output_mat[0].detach().numpy()
    np.linalg.cholesky(output_mat_i)

    # Ensure NPD operator is the negative of SPD operator
    output_n,output_mat_n = NpdMlp.forward(inputs,return_jacobian=True)
    assert torch.allclose(output,-1.0*output_n)
    assert torch.allclose(output_mat,-1.0*output_mat_n)


def test_spd_operator_scaling_forward():
    _set_seed(10)
    x_input = Variable(size=3, name="x")
    mu_input = Variable(size=2, name="mu")
    op = nnopinf.operators.SpdOperator(
        acts_on=x_input,
        depends_on=(x_input, mu_input),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
        positive=True,
    )

    x_raw = torch.randn([6, x_input.get_size()], dtype=DTYPE)
    mu_raw = torch.randn([6, mu_input.get_size()], dtype=DTYPE)
    _, mu_scaling, _ = _make_scalings(3, 2, 3)
    x_scaling = torch.tensor(2.0, dtype=DTYPE)
    x_scaling_vec = torch.full((x_input.get_size(),), x_scaling, dtype=DTYPE)
    out_scaling = torch.tensor(1.7, dtype=DTYPE)

    y_scaled = op.forward(_inputs_scaled(x_raw, mu_raw, x_scaling, mu_scaling))
    op.set_scalings({"x": x_scaling_vec, "mu": mu_scaling}, out_scaling)
    y_raw = op.forward({"x": x_raw, "mu": mu_raw})

    expected = y_scaled * out_scaling
    assert torch.allclose(y_raw, expected, rtol=1e-10, atol=1e-10)


def test_spd_operator_scaling_jacobian():
    _set_seed(11)
    x_input = Variable(size=3, name="x")
    mu_input = Variable(size=2, name="mu")
    op = nnopinf.operators.SpdOperator(
        acts_on=x_input,
        depends_on=(x_input, mu_input),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
        positive=True,
    )

    x_raw = torch.randn([4, x_input.get_size()], dtype=DTYPE)
    mu_raw = torch.randn([4, mu_input.get_size()], dtype=DTYPE)
    _, mu_scaling, _ = _make_scalings(3, 2, 3)
    x_scaling = torch.tensor(2.0, dtype=DTYPE)
    x_scaling_vec = torch.full((x_input.get_size(),), x_scaling, dtype=DTYPE)
    out_scaling = torch.tensor(1.7, dtype=DTYPE)

    _, jac_scaled = op.forward(
        _inputs_scaled(x_raw, mu_raw, x_scaling, mu_scaling), return_jacobian=True
    )
    op.set_scalings({"x": x_scaling_vec, "mu": mu_scaling}, out_scaling)
    _, jac_raw = op.forward({"x": x_raw, "mu": mu_raw}, return_jacobian=True)

    scale = (out_scaling / x_scaling).view(1, 1, 1)
    expected = jac_scaled * scale
    assert torch.allclose(jac_raw, expected, rtol=1e-10, atol=1e-10)


# ------------------
# SkewOperator tests
# ------------------

def test_skew_operator():
    _set_seed(1)
    x_input = nnopinf.Variable(size=3,name="x")
    mu_input = nnopinf.Variable(size=2,name="mu")
    SkewMlp = nnopinf.operators.SkewOperator(acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
    x = torch.randn([5,3])
    mu = torch.randn([5,2])
    inputs = {}
    inputs['x'] = x
    inputs['mu'] = mu
    output,output_mat = SkewMlp.forward(inputs,return_jacobian=True)
    # Ensure y = Ax
    assert torch.allclose(output,torch.einsum('nij,nj->ni',output_mat , x))
    # Ensure Skew
    output_mat_i = output_mat[0].detach().numpy()
    assert np.allclose(output_mat_i,-output_mat_i.transpose())


def test_skew_operator_scaling_forward():
    _set_seed(12)
    x_input = Variable(size=3, name="x")
    mu_input = Variable(size=2, name="mu")
    op = nnopinf.operators.SkewOperator(
        acts_on=x_input,
        depends_on=(x_input, mu_input),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
    )

    x_raw = torch.randn([6, x_input.get_size()], dtype=DTYPE)
    mu_raw = torch.randn([6, mu_input.get_size()], dtype=DTYPE)
    _, mu_scaling, _ = _make_scalings(3, 2, 3)
    x_scaling = torch.tensor(2.0, dtype=DTYPE)
    x_scaling_vec = torch.full((x_input.get_size(),), x_scaling, dtype=DTYPE)
    out_scaling = torch.tensor(1.7, dtype=DTYPE)

    y_scaled = op.forward(_inputs_scaled(x_raw, mu_raw, x_scaling, mu_scaling))
    op.set_scalings({"x": x_scaling_vec, "mu": mu_scaling}, out_scaling)
    y_raw = op.forward({"x": x_raw, "mu": mu_raw})

    expected = y_scaled * out_scaling
    assert torch.allclose(y_raw, expected, rtol=1e-10, atol=1e-10)


def test_skew_operator_scaling_jacobian():
    _set_seed(13)
    x_input = Variable(size=3, name="x")
    mu_input = Variable(size=2, name="mu")
    op = nnopinf.operators.SkewOperator(
        acts_on=x_input,
        depends_on=(x_input, mu_input),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
    )

    x_raw = torch.randn([4, x_input.get_size()], dtype=DTYPE)
    mu_raw = torch.randn([4, mu_input.get_size()], dtype=DTYPE)
    _, mu_scaling, _ = _make_scalings(3, 2, 3)
    x_scaling = torch.tensor(2.0, dtype=DTYPE)
    x_scaling_vec = torch.full((x_input.get_size(),), x_scaling, dtype=DTYPE)
    out_scaling = torch.tensor(1.7, dtype=DTYPE)

    _, jac_scaled = op.forward(
        _inputs_scaled(x_raw, mu_raw, x_scaling, mu_scaling), return_jacobian=True
    )
    op.set_scalings({"x": x_scaling_vec, "mu": mu_scaling}, out_scaling)
    _, jac_raw = op.forward({"x": x_raw, "mu": mu_raw}, return_jacobian=True)

    expected = jac_scaled
    assert torch.allclose(jac_raw, expected, rtol=1e-10, atol=1e-10)


# ----------------------
# StandardOperator tests
# ----------------------

def test_standard_operator():
    _set_seed(1)
    x_input = nnopinf.Variable(size=3,name="x")
    mu_input = nnopinf.Variable(size=2,name="mu")
    StandardMlp = nnopinf.operators.StandardOperator(n_outputs=6,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
    x = torch.randn([5,3])
    mu = torch.randn([5,2])
    inputs = {}
    inputs['x'] = x
    inputs['mu'] = mu
    output = StandardMlp.forward(inputs)
    assert(output.shape[1] == 6)


def test_standard_operator_scaling_forward():
    _set_seed(14)
    x_input = Variable(size=3, name="x")
    mu_input = Variable(size=2, name="mu")
    op = nnopinf.operators.StandardOperator(
        n_outputs=4,
        depends_on=(x_input, mu_input),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
    )

    x_raw = torch.randn([6, x_input.get_size()], dtype=DTYPE)
    mu_raw = torch.randn([6, mu_input.get_size()], dtype=DTYPE)
    x_scaling, mu_scaling, out_scaling = _make_scalings(3, 2, 4)

    y_scaled = op.forward(_inputs_scaled(x_raw, mu_raw, x_scaling, mu_scaling))
    op.set_scalings({"x": x_scaling, "mu": mu_scaling}, out_scaling)
    y_raw = op.forward({"x": x_raw, "mu": mu_raw})

    expected = y_scaled * out_scaling
    assert torch.allclose(y_raw, expected, rtol=1e-10, atol=1e-10)


def test_standard_operator_jacobian():
    _set_seed(3)
    x_input = nnopinf.Variable(size=3,name="x")
    mu_input = nnopinf.Variable(size=2,name="mu")
    StandardMlp = nnopinf.operators.StandardOperator(n_outputs=4,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=3)
    x = torch.randn([1,3], dtype=torch.float64)
    mu = torch.randn([1,2], dtype=torch.float64)
    inputs = {'x': x, 'mu': mu}
    output, jac = StandardMlp.forward(inputs, return_jacobian=True)

    eps = 1.0e-6
    jac_fd = torch.zeros_like(jac[0])
    for j in range(x.shape[1] + mu.shape[1]):
        dx = torch.zeros_like(x)
        dmu = torch.zeros_like(mu)
        if j < x.shape[1]:
            dx[0, j] = eps
        else:
            dmu[0, j - x.shape[1]] = eps
        y_plus = StandardMlp.forward({'x': x + dx, 'mu': mu + dmu})
        y_minus = StandardMlp.forward({'x': x - dx, 'mu': mu - dmu})
        jac_fd[:, j] = ((y_plus - y_minus) / (2.0 * eps))[0]

    assert torch.allclose(jac[0], jac_fd, rtol=1.0e-4, atol=1.0e-5)


def test_standard_operator_scaling_jacobian():
    _set_seed(15)
    x_input = Variable(size=3, name="x")
    mu_input = Variable(size=2, name="mu")
    op = nnopinf.operators.StandardOperator(
        n_outputs=4,
        depends_on=(x_input, mu_input),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
    )

    x_raw = torch.randn([4, x_input.get_size()], dtype=DTYPE)
    mu_raw = torch.randn([4, mu_input.get_size()], dtype=DTYPE)
    x_scaling, mu_scaling, out_scaling = _make_scalings(3, 2, 4)
    input_scaling = torch.cat((x_scaling, mu_scaling))

    _, jac_scaled = op.forward(
        _inputs_scaled(x_raw, mu_raw, x_scaling, mu_scaling), return_jacobian=True
    )
    op.set_scalings({"x": x_scaling, "mu": mu_scaling}, out_scaling)
    _, jac_raw = op.forward({"x": x_raw, "mu": mu_raw}, return_jacobian=True)

    row_scale = out_scaling.view(1, -1, 1)
    col_scale = (1.0 / input_scaling).view(1, 1, -1)
    expected = jac_scaled * row_scale * col_scale
    assert torch.allclose(jac_raw, expected, rtol=1e-10, atol=1e-10)


# -------------------------------
# StandardLagrangianOperator tests
# -------------------------------

def test_lagrangian_operator_jacobian_consistency():
    _set_seed(1)
    x_input = nnopinf.Variable(size=3,name="x")
    LagrangianMlp = nnopinf.operators.StandardLagrangianOperator(
        n_outputs=3,
        depends_on=(x_input,),
        n_hidden_layers=2,
        n_neurons_per_layer=4,
    )
    input_scalings = {'x': torch.tensor([2.0, 0.5, 3.0], dtype=torch.float64)}
    output_scalings = torch.tensor([1.5, 0.75, 2.0], dtype=torch.float64)
    LagrangianMlp.set_scalings(input_scalings, output_scalings)

    x = torch.randn([1, 3], dtype=torch.float64)
    inputs = {'x': x}
    grad, jac = LagrangianMlp.forward(inputs, return_jacobian=True)

    eps = 1.0e-6
    jac_fd = torch.zeros_like(jac[0])
    for j in range(x.shape[1]):
        dx = torch.zeros_like(x)
        dx[0, j] = eps
        y_plus = LagrangianMlp.forward({'x': x + dx})
        y_minus = LagrangianMlp.forward({'x': x - dx})
        jac_fd[:, j] = ((y_plus - y_minus) / (2.0 * eps))[0]

    assert torch.allclose(jac[0], jac_fd, rtol=1.0e-4, atol=1.0e-5)


def test_lagrangian_operator_scaling_forward():
    _set_seed(16)
    x_input = Variable(size=3, name="x")
    op = nnopinf.operators.StandardLagrangianOperator(
        n_outputs=3,
        depends_on=(x_input,),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
    )

    x_raw = torch.randn([6, x_input.get_size()], dtype=DTYPE)
    x_scaling, _, out_scaling = _make_scalings(3, 1, 3)

    y_scaled = op.forward({"x": x_raw / x_scaling})
    op.set_scalings({"x": x_scaling}, out_scaling)
    y_raw = op.forward({"x": x_raw})

    expected = y_scaled * (out_scaling / x_scaling)
    assert torch.allclose(y_raw, expected, rtol=1e-10, atol=1e-10)


def test_lagrangian_operator_scaling_jacobian():
    _set_seed(17)
    x_input = Variable(size=3, name="x")
    op = nnopinf.operators.StandardLagrangianOperator(
        n_outputs=3,
        depends_on=(x_input,),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
    )

    x_raw = torch.randn([4, x_input.get_size()], dtype=DTYPE)
    x_scaling, _, out_scaling = _make_scalings(3, 1, 3)

    _, jac_scaled = op.forward({"x": x_raw / x_scaling}, return_jacobian=True)
    op.set_scalings({"x": x_scaling}, out_scaling)
    _, jac_raw = op.forward({"x": x_raw}, return_jacobian=True)

    row_scale = (out_scaling / x_scaling).view(1, -1, 1)
    col_scale = (1.0 / x_scaling).view(1, 1, -1)
    expected = jac_scaled * row_scale * col_scale
    assert torch.allclose(jac_raw, expected, rtol=1e-10, atol=1e-10)


# ---------------------------
# PsdLagrangianOperator tests
# ---------------------------

def test_psd_lagrangian_operator_jacobian_consistency():
    _set_seed(2)
    x_input = nnopinf.Variable(size=3,name="x")
    mu_input = nnopinf.Variable(size=2,name="mu")
    PsdMlp = nnopinf.operators.PsdLagrangianOperator(
        acts_on=x_input,
        depends_on=(x_input, mu_input),
        n_hidden_layers=2,
        n_neurons_per_layer=4,
    )
    input_scalings = {
        'x': torch.tensor([1.5, 0.7, 2.2], dtype=torch.float64),
        'mu': torch.tensor([0.5, 1.3], dtype=torch.float64),
    }
    output_scalings = torch.tensor([1.2, 0.8, 1.5], dtype=torch.float64)
    PsdMlp.set_scalings(input_scalings, output_scalings)

    x = torch.randn([1, 3], dtype=torch.float64)
    mu = torch.randn([1, 2], dtype=torch.float64)
    inputs = {'x': x, 'mu': mu}
    grad, jac = PsdMlp.forward(inputs, return_jacobian=True)

    eps = 1.0e-6
    jac_fd = torch.zeros_like(jac[0])
    for j in range(x.shape[1]):
        dx = torch.zeros_like(x)
        dx[0, j] = eps
        y_plus = PsdMlp.forward({'x': x + dx, 'mu': mu})
        y_minus = PsdMlp.forward({'x': x - dx, 'mu': mu})
        jac_fd[:, j] = ((y_plus - y_minus) / (2.0 * eps))[0]

    assert torch.allclose(jac[0], jac_fd, rtol=1.0e-4, atol=1.0e-5)


def test_psd_lagrangian_operator_scaling_forward():
    _set_seed(18)
    x_var = Variable(size=3, name="x")
    mu_var = Variable(size=2, name="mu")

    op = PsdLagrangianOperator(
        acts_on=x_var,
        depends_on=(x_var, mu_var),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
        activation=torch.tanh,
        positive=True,
    )

    x_raw = torch.randn([6, x_var.get_size()], dtype=DTYPE)
    mu_raw = torch.randn([6, mu_var.get_size()], dtype=DTYPE)
    _, mu_scaling, _ = _make_scalings(3, 2, 3)
    x_scaling = torch.tensor(2.0, dtype=DTYPE)
    y_scaling = torch.tensor(1.7, dtype=DTYPE)

    y_scaled = op.forward(_inputs_scaled(x_raw, mu_raw, x_scaling, mu_scaling))
    op.set_scalings({"x": x_scaling, "mu": mu_scaling}, y_scaling)

    y_raw = op.forward({"x": x_raw, "mu": mu_raw})
    expected = y_scaled * y_scaling
    assert torch.allclose(y_raw, expected, rtol=1e-10, atol=1e-10)


def test_psd_lagrangian_operator_scaling_jacobian():
    _set_seed(19)
    x_var = Variable(size=3, name="x")
    mu_var = Variable(size=2, name="mu")

    op = PsdLagrangianOperator(
        acts_on=x_var,
        depends_on=(x_var, mu_var),
        n_hidden_layers=1,
        n_neurons_per_layer=4,
        activation=torch.tanh,
        positive=True,
    )

    x_raw = torch.randn([1, x_var.get_size()], dtype=DTYPE)
    mu_raw = torch.randn([1, mu_var.get_size()], dtype=DTYPE)
    _, mu_scaling, _ = _make_scalings(3, 2, 3)
    x_scaling = torch.tensor(2.0, dtype=DTYPE)
    y_scaling = torch.tensor(1.7, dtype=DTYPE)

    _, jac_scaled = op.forward(
        _inputs_scaled(x_raw, mu_raw, x_scaling, mu_scaling), return_jacobian=True
    )
    op.set_scalings({"x": x_scaling, "mu": mu_scaling}, y_scaling)

    _, jac_raw = op.forward({"x": x_raw, "mu": mu_raw}, return_jacobian=True)
    row_scale = (y_scaling / x_scaling).view(1, 1, 1)
    expected = jac_scaled * row_scale
    assert torch.allclose(jac_raw, expected, rtol=1e-10, atol=1e-10)


# ---------------------------------
# Linear affine operator empty tests
# ---------------------------------

def test_linear_affine_tensor_operator_empty_depends_on():
    _set_seed(20)
    x_var = Variable(size=3, name="x")
    op = nnopinf.operators.LinearAffineTensorOperator(
        n_outputs=3,
        acts_on=x_var,
        depends_on=(),
    )

    x = torch.randn([6, x_var.get_size()], dtype=DTYPE)
    y, jac = op.forward({"x": x}, return_jacobian=True)

    assert y.shape == (6, 3)
    assert jac.shape == (6, 3, 3)
    assert torch.allclose(y, torch.einsum("bij,bj->bi", jac, x), rtol=1e-12, atol=1e-12)

    op.set_scalings(
        {"x": torch.tensor([1.4, 1.8, 2.2], dtype=DTYPE)},
        torch.tensor([1.1, 1.3, 1.7], dtype=DTYPE),
    )
    y_scaled = op.forward({"x": x})
    assert y_scaled.shape == y.shape


def test_linear_affine_skew_tensor_operator_empty_depends_on():
    _set_seed(21)
    x_var = Variable(size=4, name="x")
    op = nnopinf.operators.LinearAffineSkewTensorOperator(
        acts_on=x_var,
        depends_on=(),
        skew=True,
    )

    x = torch.randn([5, x_var.get_size()], dtype=DTYPE)
    y = op.forward({"x": x})

    assert y.shape == x.shape
    # For skew-symmetric operators, x^T A x = 0.
    energy = torch.sum(x * y, dim=1)
    assert torch.allclose(energy, torch.zeros_like(energy), rtol=1e-10, atol=1e-10)

    op.set_scalings(
        {"x": torch.tensor([1.2, 1.4, 1.6, 1.8], dtype=DTYPE)},
        torch.tensor([1.1, 1.2, 1.3, 1.4], dtype=DTYPE),
    )
    y_scaled = op.forward({"x": x})
    assert y_scaled.shape == y.shape


def test_linear_affine_spd_tensor_operator_empty_depends_on_sign():
    _set_seed(22)
    x_var = Variable(size=3, name="x")
    op_pos = nnopinf.operators.LinearAffineSpdTensorOperator(
        acts_on=x_var,
        depends_on=(),
        positive=True,
    )
    _set_seed(22)
    op_neg = nnopinf.operators.LinearAffineSpdTensorOperator(
        acts_on=x_var,
        depends_on=(),
        positive=False,
    )

    x = torch.randn([7, x_var.get_size()], dtype=DTYPE)
    y_pos = op_pos.forward({"x": x})
    y_neg = op_neg.forward({"x": x})
    assert torch.allclose(y_pos, -y_neg, rtol=1e-12, atol=1e-12)

    x_scaling = torch.tensor([1.3, 1.7, 2.1], dtype=DTYPE)
    y_scaling = torch.tensor([1.1, 1.4, 1.8], dtype=DTYPE)
    op_pos.set_scalings({"x": x_scaling}, y_scaling)
    op_neg.set_scalings({"x": x_scaling}, y_scaling)
    y_pos_scaled = op_pos.forward({"x": x})
    y_neg_scaled = op_neg.forward({"x": x})
    assert torch.allclose(y_pos_scaled, -y_neg_scaled, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
  test_spd_operator_scaling_forward()
