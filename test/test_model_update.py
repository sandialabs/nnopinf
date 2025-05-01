import torch
import nnopinf.operators as operators
import nnopinf.models as models
import numpy as np

if __name__=='__main__':
    rom_dim_coarse = 5
    rom_dim_fine = 10

    n_hidden_layers = 5
    n_neurons_per_layer = 9

    NpdMlpCoarse = operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,5,5)
    SkewMlpCoarse = operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,5,5)
    MyCompositeOperatorCoarse = operators.CompositeOperator([NpdMlpCoarse,SkewMlpCoarse])

    NpdMlpFine = operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,10,10)
    SkewMlpFine = operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,10,10)
    MyCompositeOperatorFine = operators.CompositeOperator([NpdMlpFine,SkewMlpFine])

    coarse_operators =  [  models.WrappedOperatorForModel(operator=MyCompositeOperatorCoarse,inputs=("x"),name='stiffness') ]
    fine_operators =    [  models.WrappedOperatorForModel(operator=MyCompositeOperatorFine,inputs=("x"),name='stiffness') ]

          
    my_model_coarse = models.OpInfModel( coarse_operators )
    my_model_fine = models.OpInfModel( fine_operators )
    my_model_fine.hierarchical_update(my_model_coarse)

    sample_state = np.random.normal(size=10)
    sample_state[5::] = 0.
    sample_state = torch.tensor(sample_state[None])
    inputs_coarse = {'x':sample_state[...,0:5]}
    inputs_fine = {'x':sample_state}
    outputs_coarse = my_model_coarse.forward(inputs_coarse)
    outputs_fine = my_model_fine.forward(inputs_fine)

    assert np.allclose( outputs_coarse.detach().numpy(),outputs_fine.detach().numpy()[...,0:5])

