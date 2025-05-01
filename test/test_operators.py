import nnopinf
import nnopinf.operators

import torch
import numpy as np

if __name__ == "__main__":
    n_hidden_layers = 2
    n_neurons_per_layer = 5
    n_inputs = 5
    n_outputs = 5
    inputs = torch.tensor(np.random.normal(size=(10,n_inputs)))
    parameters = torch.tensor(np.random.normal(size=(10,0)))
    
      
    StandardMlp = nnopinf.operators.StandardOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SpdMlp = nnopinf.operators.SpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    NpdMlp = nnopinf.operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SkewMlp = nnopinf.operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    MatrixMlp = nnopinf.operators.MatrixOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,(n_outputs,n_inputs))
   
    ops = [SpdMlp,StandardMlp,SkewMlp,MatrixMlp]
    CompositeMlp = nnopinf.operators.CompositeOperator(ops)
    inputs = torch.cat((inputs,parameters),1)
    r1 = StandardMlp.forward(inputs)
    r2 = SpdMlp.forward(inputs)
    r3 = SkewMlp.forward(inputs)
    r4 = MatrixMlp.forward(inputs)
    r5 = CompositeMlp(inputs)
    assert np.allclose( r5.detach().numpy(), (r1 + r2 + r3 + r4).detach().numpy())

    ## Now test scaling
    for operator in ops:
      input_scalings = torch.tensor(np.random.normal(size=(n_inputs)))
      output_scalings = torch.tensor(np.random.normal(size=(n_outputs)))
    
      inputs_scaled = inputs/input_scalings[None] 
      output = operator.forward(inputs_scaled)*output_scalings[None]

      operator.set_scalings(input_scalings,output_scalings)
      output2 = operator.forward(inputs)
      assert np.allclose(operator.input_scaling_,input_scalings)
      assert np.allclose(operator.output_scaling_,output_scalings)
      assert np.allclose(output.detach().numpy(),output2.detach().numpy())


    ## Test heirarchical update
    SpdMlp = nnopinf.operators.SpdOperator(n_hidden_layers,n_neurons_per_layer,5,5)
    SpdMlpFine = nnopinf.operators.SpdOperator(n_hidden_layers,n_neurons_per_layer,10,10)
    SpdMlpFine.heirarchical_update(SpdMlp)
  

