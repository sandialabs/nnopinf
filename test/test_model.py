import torch
import nnopinf.operators as operators
import nnopinf.models as models
import numpy as np

if __name__=='__main__':
    rom_dim = 10
    exo_input_dim = 4

    n_params = 2
    n_hidden_layers = 2
    n_neurons_per_layer = 5

    ## Design operators for the state
    NpdMlp = operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SkewMlp = operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)


    ## Design operators for the forcing
    StandardMlp = operators.StandardOperator(n_hidden_layers,n_neurons_per_layer,n_params,rom_dim)
    operators =  [  models.WrappedOperatorForModel(operator=MyCompositeOperator,inputs=("x","mu",),name='stiffness'),  models.WrappedOperatorForModel(operator=StandardMlp,inputs=("mu",),name='input') ]
 
          
    my_model = models.OpInfModel( operators )
          
    sample_state = torch.tensor( np.ones((1,rom_dim)))
    sample_parameters =  torch.tensor( np.ones((1,n_params)))
    inputs = {"x":sample_state,"mu":sample_parameters}             
    outputs = my_model.forward(inputs)
    outputs_check = MyCompositeOperator.forward( torch.cat((sample_state,sample_parameters),1))
    outputs_check +=  StandardMlp.forward(sample_parameters)

    assert np.allclose( outputs.detach().numpy(),outputs_check.detach().numpy())

    x_scalings = 0.1*torch.tensor(np.ones(rom_dim))
    mu_scalings = .02*torch.tensor(np.ones(n_params))

    inputs_scaled = {"x":sample_state/x_scalings[None],"mu":sample_parameters/mu_scalings[None]}
    outputs_scaled = my_model.forward(inputs_scaled)
    outputs_unscaled = outputs_scaled*x_scalings[None]

    input_scalings = {}
    input_scalings['x'] = x_scalings
    input_scalings['mu'] = mu_scalings
    my_model.set_scalings(input_scalings,x_scalings)

    new_outputs = my_model.forward(inputs)
    assert np.allclose(new_outputs.detach().numpy(),outputs_unscaled.detach().numpy())    

