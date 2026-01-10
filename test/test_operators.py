import nnopinf
import nnopinf.operators
import torch
import numpy as np


def test_spd_operator():
    torch.manual_seed(1)
    np.random.seed(1)

    x_input = nnopinf.Variable(size=3,name="x")
    mu_input = nnopinf.Variable(size=2,name="mu")
    NpdMlp = nnopinf.operators.SpdOperator(acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2,positive=False)
 
    torch.manual_seed(1)
    np.random.seed(1)

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

 
def test_skew_operator():
    torch.manual_seed(1)
    np.random.seed(1)
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

def test_standard_operator():
    torch.manual_seed(1)
    np.random.seed(1)
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

