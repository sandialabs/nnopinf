import numpy as np
import torch
import nnopinf
import nnopinf.operators as operators
import torch.nn.functional as F
import torch.nn as nn
import os
torch.set_default_dtype(torch.float64)


class Model(nn.Module):
    """
    Test
    """

    def __init__(self,operators):
        super(Model, self).__init__()
        # Wrap operators into a ModuleList so pytorch sees them for training
        self.operators_ = operators
        self.operator_modules_ = []
        for i in range(0,len(operators)):
            self.operator_modules_.append(operators[i])
        self.operator_modules_ = nn.ModuleList(self.operator_modules_)
        self.scalings_set_ = False

    def forward(self,inputs,return_stiffness=False):
        output = 0
        stiffness = 0
        for i in range(0,len(self.operators_)):
            if isinstance(self.operators_[i],nnopinf.operators.VectorOffsetOperator):
              output_l = self.operators_[i].forward(inputs,return_jacobian=False)
              output += output_l
            else:
              if return_stiffness:
                output_l,stiffness_l = self.operators_[i].forward(inputs,return_stiffness=return_stiffness)
                output += output_l
                stiffness += stiffness_l 
              else:
                output_l = self.operators_[i].forward(inputs)
                output += output_l

        if return_stiffness:
          return output,stiffness
        else:
          return output

    def save_operators(self,output_directory):
        if os.path.isdir(output_directory):
            pass
        else:
            os.makedirs(output_directory)

        for operator in self.operators_:
          torch.save(operator,output_directory + '/' + operator.name_ + '.pt')

    def set_scalings(self,input_scalings_dict,output_scalings):
        self.scalings_set_ = True
        for i in range(0,len(self.operators_)):
            input_scalings = None
            self.operators_[i].set_scalings(input_scalings_dict,output_scalings)


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

