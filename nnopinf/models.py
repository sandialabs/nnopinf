import numpy as np
import torch
import nnopinf.operators as operators
import torch.nn.functional as F
import torch.nn as nn
import os
torch.set_default_dtype(torch.float64)


class OpInfModel(nn.Module):
    def __init__(self,wrapped_operators):
        super(OpInfModel, self).__init__()
        self.wrapped_operators_ = wrapped_operators

        # Wrap operators into a ModuleList so pytorch sees them for training
        self.operator_modules_ = []
        for i in range(0,len(self.wrapped_operators_)):
            self.operator_modules_.append(self.wrapped_operators_[i].operator_)
        self.operator_modules_ = nn.ModuleList(self.operator_modules_)


    def forward(self,inputs):
        output = 0
        for i in range(0,len(self.wrapped_operators_)):
            operator_inputs = None
            for input_arg in self.wrapped_operators_[i].inputs_:
                if operator_inputs is None:
                  operator_inputs = inputs[input_arg]
                else:
                  operator_inputs = torch.cat( (operator_inputs,inputs[input_arg]),1)
            output += self.operator_modules_[i].forward(operator_inputs)
      
        return output

    def save_operators(self,output_directory):
        if os.path.isdir(output_directory):
            pass
        else:
            os.makedirs(output_directory)

        for wrapped_operator in self.wrapped_operators_:
          torch.save(wrapped_operator.operator_,output_directory + '/' + wrapped_operator.name_ + '.pt')

    def set_scalings(self,input_scalings_dict,output_scalings):
        for i in range(0,len(self.wrapped_operators_)):
            input_scalings = None
            for input_arg in self.wrapped_operators_[i].inputs_:
                if input_scalings is None:
                  input_scalings = input_scalings_dict[input_arg]
                else:
                  input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
            self.wrapped_operators_[i].operator_.set_scalings(input_scalings,output_scalings)

    def hierarchical_update(self,input_model):
        for i in range(0,len(self.wrapped_operators_)):
            self.wrapped_operators_[i].operator_.hierarchical_update(input_model.wrapped_operators_[i].operator_)


    def remove_spectral_norm(self):
        for i in range(0,len(self.wrapped_operators_)):
            self.wrapped_operators_[i].operator_.remove_spectral_norm()

class WrappedOperatorForModel:
    def __init__(self,operator,inputs,name):
        self.operator_ = operator
        self.inputs_ = inputs 
        self.name_ = name



if __name__=='__main__':
    rom_dim = 10
    exo_input_dim = 4

    n_params = 2
    n_hidden_layers = 2
    n_neurons_per_layer = 5
    n_inputs = rom_dim + n_params
    n_outputs = rom_dim

    ## Design operators for the state
    NpdMlp = operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SkewMlp = operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    MyCompositeOperator = operators.CompositeOperator([NpdMlp,SkewMlp])


    ## Design operators for the forcing
    StandardMlp = operators.StandardOperator(n_hidden_layers,n_neurons_per_layer,n_params,rom_dim)

    WrappedCompositeOperator = WrappedOperatorForModel(operator=MyCompositeOperator,inputs=("x","mu"),name="stiffness")
    WrappedParametricOperator = WrappedOperatorForModel(operator=StandardMlp,inputs=("mu",),name="forcing")

 

    my_model = OpInfModel( [WrappedCompositeOperator,WrappedParametricOperator] )
 
    sample_state = torch.tensor( np.ones((1,rom_dim)))
    sample_parameters =  torch.tensor( np.ones((1,n_params)))
    inputs = {"x":sample_state,"mu":sample_parameters}
    outputs = my_model.forward(inputs)
    outputs_check = MyCompositeOperator.forward( torch.cat((sample_state,sample_parameters),1))
    outputs_check +=  StandardMlp.forward(sample_parameters)
    x_scalings = torch.tensor(np.ones(rom_dim))
    mu_scalings = torch.tensor(np.ones(n_params))
    scalings = {}
    scalings['x'] = x_scalings
    scalings['mu'] = mu_scalings

    model.set_scalings(scalings)
    assert np.allclose( outputs.detach().numpy(),outputs_check.detach().numpy())

