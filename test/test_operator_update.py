import nnopinf
import nnopinf.operators

import torch
import numpy as np

if __name__ == "__main__":
    n_hidden_layers = 2
    n_neurons_per_layer = 5
    inputs = np.random.normal(size=10)
    inputs[5::] = 0.
    inputs = torch.tensor(inputs)
 
    ## Test hierarchical update
    SpdMlp = nnopinf.operators.SpdOperator(n_hidden_layers,n_neurons_per_layer,5,5)
    SpdMlpFine = nnopinf.operators.SpdOperator(n_hidden_layers,n_neurons_per_layer,10,10)
    SpdMlpFine.hierarchical_update(SpdMlp)
  
    coarse_outputs,coarse_stiffness = SpdMlp.forward(inputs[None,0:5],True)
    fine_outputs,fine_stiffness = SpdMlpFine.forward(inputs[None],True)
    assert np.allclose(coarse_outputs.detach().numpy()[0],fine_outputs.detach().numpy()[0,0:5])
    assert np.allclose(coarse_stiffness.detach().numpy()[0],fine_stiffness.detach().numpy()[0,0:5,0:5])
    NpdMlp = nnopinf.operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,5,5)
    NpdMlpFine = nnopinf.operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,10,10)
    NpdMlpFine.hierarchical_update(NpdMlp)
  
    coarse_outputs,coarse_stiffness = NpdMlp.forward(inputs[None,0:5],True)
    fine_outputs,fine_stiffness = NpdMlpFine.forward(inputs[None],True)
    assert np.allclose(coarse_outputs.detach().numpy()[0],fine_outputs.detach().numpy()[0,0:5])
    assert np.allclose(coarse_stiffness.detach().numpy()[0],fine_stiffness.detach().numpy()[0,0:5,0:5])

    SkewMlp = nnopinf.operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,5,5)
    SkewMlpFine = nnopinf.operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,10,10)
    SkewMlpFine.hierarchical_update(SkewMlp)
  
    coarse_outputs,coarse_stiffness = SkewMlp.forward(inputs[None,0:5],True)
    fine_outputs,fine_stiffness = SkewMlpFine.forward(inputs[None],True)
    assert np.allclose(coarse_outputs.detach().numpy()[0],fine_outputs.detach().numpy()[0,0:5])
    assert np.allclose(coarse_stiffness.detach().numpy()[0],fine_stiffness.detach().numpy()[0,0:5,0:5])

    CompositeMlp = nnopinf.operators.CompositeOperator([NpdMlp,SkewMlp])
    CompositeMlpFine = nnopinf.operators.CompositeOperator([NpdMlpFine,SkewMlpFine])
    CompositeMlpFine.hierarchical_update(CompositeMlp)
  
    coarse_outputs,coarse_stiffness = CompositeMlp.forward(inputs[None,0:5],True)
    fine_outputs,fine_stiffness = CompositeMlpFine.forward(inputs[None],True)
    assert np.allclose(coarse_outputs.detach().numpy()[0],fine_outputs.detach().numpy()[0,0:5])
    assert np.allclose(coarse_stiffness.detach().numpy()[0],fine_stiffness.detach().numpy()[0,0:5,0:5])


    StandardMlp = nnopinf.operators.StandardOperator(n_hidden_layers,n_neurons_per_layer,5,5)
    StandardMlpFine = nnopinf.operators.StandardOperator(n_hidden_layers,n_neurons_per_layer,10,10)
    StandardMlpFine.hierarchical_update(StandardMlp)
  
    coarse_outputs = StandardMlp.forward(inputs[None,0:5])
    fine_outputs = StandardMlpFine.forward(inputs[None])
    assert np.allclose(coarse_outputs.detach().numpy()[0],fine_outputs.detach().numpy()[0,0:5])
    #assert np.allclose(coarse_stiffness.detach().numpy()[0],fine_stiffness.detach().numpy()[0,0:5,0:5])

    MatrixMlp = nnopinf.operators.MatrixOperator(n_hidden_layers,n_neurons_per_layer,3,(5,3))
    MatrixMlpFine = nnopinf.operators.MatrixOperator(n_hidden_layers,n_neurons_per_layer,3,(10,3))
    MatrixMlpFine.hierarchical_update(MatrixMlp)
  
    coarse_outputs,coarse_stiffness = MatrixMlp.forward(inputs[None,0:3],True)
    fine_outputs,fine_stiffness = MatrixMlpFine.forward(inputs[None,0:3],True)
    assert np.allclose(coarse_outputs.detach().numpy()[0],fine_outputs.detach().numpy()[0,0:5])
    assert np.allclose(coarse_stiffness.detach().numpy()[0],fine_stiffness.detach().numpy()[0,0:5,0:5])
