import torch
import nnopinf
import nnopinf.training
import nnopinf.operators as operators
import nnopinf.models as models
import numpy as np

if __name__ == "__main__":                                                                                                                   

    rom_dim = 10                                                                                                                             
                                                                                                                                             
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

    WrappedCompositeOperator = models.WrappedOperatorForModel(operator=MyCompositeOperator,inputs=("x","mu"),name="stiffness")
    WrappedParametricOperator = models.WrappedOperatorForModel(operator=StandardMlp,inputs=("mu",),name="forcing")
                                                                                                                                             
    my_model = models.OpInfModel( [WrappedCompositeOperator,WrappedParametricOperator] )
 
    #Construct training data
    n_samples = 50
    sample_state = np.random.normal(size=(n_samples,rom_dim))                                                                                
    sample_parameters =  np.random.normal(size=(n_samples,n_params))                                                                         

    inputs = {"x":sample_state,"mu":sample_parameters}

    sample_response =  np.random.normal(size=(n_samples,rom_dim))                                                                            
    training_settings = nnopinf.training.get_default_settings()
    training_settings['num-epochs'] = 5
    nnopinf.training.train(my_model,inputs,sample_response,training_settings)  
