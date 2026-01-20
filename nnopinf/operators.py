import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import nnopinf.variables
from typing import Protocol

torch.set_default_dtype(torch.float64)

def inputs_to_tensor(inputs,names_to_collect):
    operator_inputs = []
    for input_name in names_to_collect:
      operator_inputs.append(inputs[input_name])
    if len(operator_inputs) == 0:
      return torch.zeros((0,0))
    return torch.cat(operator_inputs, dim=1)

def create_layers(input_size,output_size,n_hidden_layers,n_neurons_per_layer):
      dim = np.zeros(n_hidden_layers+2,dtype='int')
      dim[0] = input_size 
      for i in range(1,n_hidden_layers+1):
       dim[i] = n_neurons_per_layer
      dim[-1] = output_size 
      input_dim = dim[0:-1]
      output_dim = dim[1::]
      forward_list = []
      for i in range(0,n_hidden_layers+1):
        forward_list.append(nn.Linear(input_dim[i], output_dim[i],bias=True))
      return nn.ModuleList(forward_list)
    


class Operator(Protocol):
    '''
    Protocol for operator class.
    '''

    def forward(self,inputs,return_jacobian):
      """
      Forward pass of operator 

      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool 
          If the method returns the (approximate) Jacobian, A. 
      """ 

    def set_scalings(self,input_scaling,output_scaling):
      pass 


class CompositeOperator(nn.Module):
    """
    :math:`f: (f_1,\ldots,f_K) \mapsto \sum_{i=1}^K f_k` 

    Constructs an operator composed of other NN-OpInf operators 

    Parameters
    ----------
    state_operators: list of nnopinf.operators.Operator 
        List of individual operators 
 
    name : string 
        Operator name. Used when saving to file 

    Examples
    --------
    >>> import nnopinf
    >>> import nnopinf.operators
    >>> x_input = nnopinf.Variable(size=3,name="x")
    >>> mu_input = nnopinf.Variable(size=2,name="mu")
    >>> MatrixMlp = nnopinf.operators.MatrixOperator(n_outputs=5,acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
    >>> MatrixMlp2 = nnopinf.operators.MatrixOperator(n_outputs=5,acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
    >>> CompositeMlp = nnopinf.operators.CompositeOperator([MatrixMlp,MatrixMlp2])
    """

    def __init__(self,state_operators,name='CompositeOperator'):
        super(CompositeOperator, self).__init__()
        self.state_operators =  nn.ModuleList(state_operators)
        self.name_ = name

    def forward(self,inputs,return_jacobian=False):
        if return_jacobian:
            result,stiffness = self.state_operators[0].forward(inputs,return_jacobian=True)
            for state_operator in self.state_operators[1::]:
                t1,t2 = state_operator.forward(inputs,return_jacobian=True)
                result += t1
                stiffness += t2
            return result,stiffness
        else:
            result = self.state_operators[0].forward(inputs)
            for state_operator in self.state_operators[1::]:
                result += state_operator.forward(inputs)
            return result 

    def set_scalings(self,input_scaling,output_scaling):
        for operator in self.state_operators:
            operator.set_scalings(input_scaling,output_scaling)

class VectorOffsetOperator(nn.Module):
    """
    :math:`f \in \mathbb{R}^M` 

    Constructs a constant vector operator, :math:`f \in \mathbb{R}^M` 
 
    Parameters
    ----------
    n_outputs: int 
        Output dimension of the operator, i.e., ``M`` in the above description 
 
    name : string 
        Operator name. Used when saving to file 

    """

    def __init__(self,n_outputs,name='VectorOffsetOperator'):
        super(VectorOffsetOperator, self).__init__()
        self.name_ = name
        self.n_outputs_ = n_outputs 
        network_output_size = self.n_outputs_ 
        self.vec = nn.Parameter(torch.randn(n_outputs))

    def forward(self,inputs,return_jacobian):
      """ 
      Forward pass of operator 

      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool 
          If the method returns the (approximate) Jacobian, A. 
  
      """
      if return_jacobian: 
        return self.vec,torch.zeros((self.n_outputs,0))
      else:
        return self.vec 

    def set_scalings(self,input_scalings_dict,output_scaling):
       with torch.no_grad():
         self.vec[:] *= output_scaling




class StandardOperator(nn.Module):
    """
    :math:`f: v \mapsto f(v)` 

    Constructs an operator :math:`f: v \mapsto f(v),` with a dense neural network for :math:`v \in \mathbb{R}^K`, and :math:`f(v) \in \mathbb{R}^{M}`.

    .. note::

       - The output dimension does not need to match the input dimension. 
       - There is no "acts on" input as we are not inferring a matrix operator.

    Parameters
    ----------
    n_outputs: int
        Output dimension of the operator, i.e., ``M`` in the above description

    depends_on: tuple of nnopinf.Variable 
        The variables the operator depends on, i.e., the ``v`` in ``f(v)``

    n_hidden_layers : int 
        Number of hidden layers in the network 

    n_neurons_per_layer : int 
        Number of nuerons in each hidden layer

    activation : PyTorch activation function (e.g., torch.nn.functional.relu)
        Activation function used at each layer
 
    name : string 
        Operator name. Used when saving to file 

    Examples
    --------
    >>> import nnopinf
    >>> import nnopinf.operators
    >>> x_input = nnopinf.Variable(size=3,name="x")
    >>> mu_input = nnopinf.Variable(size=2,name="mu")
    >>> StandardMlp = nnopinf.operators.StandardOperator(n_outputs=5,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
    """

    def __init__(self,n_outputs,depends_on,n_hidden_layers,n_neurons_per_layer,activation=F.relu,name='StandardOperator'):
        super(StandardOperator, self).__init__()
        self.name_ = name
        self.n_outputs_ = n_outputs 
        self.depends_on_names_ = []
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())

        network_output_size = self.n_outputs_ 
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        n_inputs = 0
        for operator_input in depends_on:
          n_inputs += operator_input.get_size()
        self.n_inputs_ = n_inputs
        self.forward_list = create_layers(n_inputs,network_output_size,n_hidden_layers,n_neurons_per_layer)

        self.activation = activation
        self.input_scaling_ = torch.ones(n_inputs)
        self.output_scaling_ = torch.ones(n_outputs)

    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 

      .. warning:: return_jacobian is currently not implemented 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool 
          If the method returns the (approximate) Jacobian, A. 
  
      Examples
      --------
      >>> import nnopinf
      >>> import nnopinf.operators
      >>> import numpy as np
      >>> x_input = nnopinf.Variable(size=3,name="x")
      >>> mu_input = nnopinf.Variable(size=2,name="mu")
      >>> StandardMlp = nnopinf.operators.StandardOperator(n_outputs = 5,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
      >>> inputs = {}
      >>> inputs['x'] = np.random.normal(3)
      >>> inputs['mu'] = np.random.normal(2)
      >>> f = MatrixMlp.forward(inputs)
      """

      if return_jacobian == True:
        print("Error, return_jacobian not implemented for StandardOperator")
      else: 
        y = inputs_to_tensor(inputs,self.depends_on_names_) 
        for i in range(0,self.num_layers-1):
          y = self.activation(self.forward_list[i](y))
        result = self.forward_list[-1](y)
      return result[:,:]


    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      input_scalings = None
      for input_arg in self.depends_on_names_:
        if input_scalings is None:
          input_scalings = input_scalings_dict[input_arg]
        else:
          input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      
      # Update initial layer weights
      initial_layer = self.forward_list[0]
      initial_layer.weight[:] = initial_layer.weight[:] @ torch.eye(self.n_inputs_)/input_scalings
       
      # Update final layer weights
      final_layer = self.forward_list[-1]
      final_layer.weight[:] =  (output_scaling * torch.eye(self.n_inputs_)) @ final_layer.weight[:]
      final_layer.bias[:] =  output_scaling * final_layer.bias[:]




class SpdOperator(nn.Module):
    """
    :math:`f: (v,x) \mapsto L(v)L(v)^T x` 

    Constructs an SPD (or NPD) operator :math:`f: (v,x) \mapsto L(v)L(v)^Tx = A(v)x` such that :math:`x^T A(v) x >= 0`

    Parameters
    ----------
    acts_on : nnopinf.Variable
        The state the operators acts on, i.e., the ``x`` in ``A(v) x``

    depends_on: tuple of nnopinf.Variable 
        The variables the operator depends on, i.e., the ``v`` in ``A(v) x``

    n_hidden_layers : int 
        Number of hidden layers in the network 

    n_neurons_per_layer : int 
        Number of nuerons in each hidden layer

    activation : PyTorch activation function (e.g., torch.nn.functional.relu)
        Activation function used at each layer
 
    positive : bool 
        If operator is SPD or NPD 

    name : string 
        Operator name. Used when saving to file 

    Examples
    --------
    >>> import nnopinf
    >>> import nnopinf.operators
    >>> x_input = nnopinf.Variable(size=3,name="x")
    >>> mu_input = nnopinf.Variable(size=2,name="mu")
    >>> NpdMlp = nnopinf.operators.SpdOperator(acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2,positive=False)
    """

    def __init__(self,acts_on,depends_on,n_hidden_layers,n_neurons_per_layer,activation=F.relu, positive=True,name='SpdOperator'):
        super(SpdOperator, self).__init__()
        self.name_ = name
        forward_list = []

        self.positive_ = positive 
        self.n_outputs_ = acts_on.get_size()

        self.acts_on_name_ = acts_on.get_name()
        self.depends_on_names_ = []
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())

        idx = np.tril_indices(self.n_outputs_)
        self.idx = idx

        network_output_size = idx[0].size
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        n_inputs = 0
        for operator_input in depends_on:
          n_inputs += operator_input.get_size()

        self.n_inputs_ = n_inputs

        if self.positive_:
          self.scale_ = 1.0
        else:
          self.scale_ = -1.0
        self.forward_list = create_layers(n_inputs,network_output_size,n_hidden_layers,n_neurons_per_layer)
        self.activation = activation
        self.scaling_mat_ = torch.eye(self.n_outputs_)

    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool 
          If the method returns the (approximate) Jacobian, A. 
  
      Examples
      --------
      >>> import nnopinf
      >>> import nnopinf.operators
      >>> import numpy as np
      >>> x_input = nnopinf.Variable(size=3,name="x")
      >>> mu_input = nnopinf.Variable(size=2,name="mu")
      >>> NpdMlp = nnopinf.operators.SpdOperator(acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2,positive=False)
      >>> inputs = {}
      >>> inputs['x'] = np.random.normal(3)
      >>> inputs['mu'] = np.random.normal(2)
      >>> Av,A = NpdMlp.forward(inputs,True)
      """

      y = inputs_to_tensor(inputs,self.depends_on_names_)
      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))

      y = self.forward_list[-1](y)
      K = torch.zeros(y.shape[0],self.n_outputs_,self.n_outputs_)
      K[:,self.idx[0],self.idx[1]] = y[:,0:self.idx[0].size]
      KT = torch.transpose(K,2,1)
      K = torch.einsum('ijk,ikl->ijl',K,KT)
      state = inputs[self.acts_on_name_] 
      state = torch.einsum('ij,nj->ni',self.scaling_mat_ , state)
      result = torch.einsum('ijk,ik->ij',K,state )
      
      if return_jacobian:
          return result*self.scale_, K*self.scale_
      else:
          return result*self.scale_

    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      input_scalings = None
      for input_arg in self.depends_on_names_:
        if input_scalings is None:
          input_scalings = input_scalings_dict[input_arg]
        else:
          input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      
      # Update initial layer weights
      initial_layer = self.forward_list[0]
      initial_layer.weight[:] = initial_layer.weight[:] @ torch.eye(self.n_inputs_)/input_scalings
      self.scaling_mat_[:] = torch.eye(self.n_outputs_) / input_scalings_dict[self.acts_on_name_] * output_scaling


class SkewOperator(nn.Module):
    """
    :math:`f: (v,x) \mapsto [S(v) - S(v)^T] x` 

    Constructs an operator :math:`f: (v,x) \mapsto [S(v) - S(v)^T] x = A(v)x` such that :math:`A(v) = -A(v)^T`

    Parameters
    ----------
    acts_on : nnopinf.Variable
        The state the operators acts on, i.e., the ``x`` in ``A(v) x``

    depends_on: tuple of nnopinf.Variable 
        The variables the operator depends on, i.e., the ``v`` in ``A(v) x``

    n_hidden_layers : int 
        Number of hidden layers in the network 

    n_neurons_per_layer : int 
        Number of nuerons in each hidden layer

    activation : PyTorch activation function (e.g., torch.nn.functional.relu)
        Activation function used at each layer
 
    name : string 
        Operator name. Used when saving to file 

    Examples
    --------
    >>> import nnopinf
    >>> import nnopinf.operators
    >>> x_input = nnopinf.Variable(size=3,name="x")
    >>> mu_input = nnopinf.Variable(size=2,name="mu")
    >>> SkewMlp = nnopinf.operators.SkewOperator(acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
    """

    def __init__(self,acts_on,depends_on,n_hidden_layers,n_neurons_per_layer,activation=F.relu, name='SkewOperator'):
        super(SkewOperator, self).__init__()
        self.name_ = name
        forward_list = []

        self.n_outputs_ = acts_on.get_size()
        idx = np.tril_indices(self.n_outputs_)
        self.idx = idx

        self.acts_on_name_ = acts_on.get_name()
        self.depends_on_names_ = []
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())


        network_output_size = idx[0].size
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1
        self.scaling_mat_ = torch.eye(self.n_outputs_)

        n_inputs = 0
        for operator_input in depends_on:
          n_inputs += operator_input.get_size()
        self.n_inputs_ = n_inputs
        self.forward_list = create_layers(n_inputs,network_output_size,n_hidden_layers,n_neurons_per_layer)

        self.activation = activation  
        self.scaling_mat_ = torch.eye(self.n_outputs_)     


    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool 
          If the method returns the (approximate) Jacobian, A. 
  
      Examples
      --------
      >>> import nnopinf
      >>> import nnopinf.operators
      >>> import numpy as np
      >>> x_input = nnopinf.Variable(size=3,name="x")
      >>> mu_input = nnopinf.Variable(size=2,name="mu")
      >>> SkewMlp = nnopinf.operators.SkewOperator(acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
      >>> inputs = {}
      >>> inputs['x'] = np.random.normal(3)
      >>> inputs['mu'] = np.random.normal(2)
      >>> Av,A = SkewMlp.forward(inputs,True)
      """

      y = inputs_to_tensor(inputs,self.depends_on_names_)
      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))

      y = self.forward_list[-1](y)

      S = torch.zeros(y.shape[0],self.n_outputs_,self.n_outputs_)

      S[:,self.idx[0],self.idx[1]] = y[:,0:self.idx[0].size]

      ST = torch.transpose(S,2,1)
      R = S - ST 
      # Apply scaling to R so matrix has correct scaling

      state = inputs[self.acts_on_name_] 
      state = torch.einsum('ij,nj->ni',self.scaling_mat_ , state)
      result = torch.einsum('ijk,ik->ij',R,state)
      if return_jacobian:
          return result[:,:],R
      else:
          return result[:,:]


    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      input_scalings = None
      for input_arg in self.depends_on_names_:
        if input_scalings is None:
          input_scalings = input_scalings_dict[input_arg]
        else:
          input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      
      # Update initial layer weights
      initial_layer = self.forward_list[0]
      initial_layer.weight[:] = initial_layer.weight[:] @ torch.eye(self.n_inputs_)/input_scalings
      self.scaling_mat_[:] = torch.eye(self.n_outputs_) / input_scalings_dict[self.acts_on_name_] * output_scaling




class MatrixOperator(nn.Module):
    """
    :math:`f: (v,x) \mapsto A(v)x` 

    Constructs an operator :math:`f: A(v)x` for :math:`x \in \mathbb{R}^K, v \in \mathbb{R}^{N}`, and :math:`A \in \mathbb{R}^{M \times K}`
    Note that the Matrix :math:`A` does not need to be square

    Parameters
    ----------
    n_outputs: int
        Output dimension of the operator, i.e., ``M`` in the above description

    acts_on : nnopinf.Variable
        The state the operators acts on, i.e., the :math:`x` in :math:`A(v) x`

    depends_on: tuple of nnopinf.Variable 
        The variables the operator depends on, i.e., the :math:`v` in :math:`A(v) x`

    n_hidden_layers : int 
        Number of hidden layers in the network 

    n_neurons_per_layer : int 
        Number of nuerons in each hidden layer

    activation : PyTorch activation function (e.g., torch.nn.functional.relu)
        Activation function used at each layer
 
    name : string 
        Operator name. Used when saving to file 

    Examples
    --------
    >>> import nnopinf
    >>> import nnopinf.operators
    >>> x_input = nnopinf.Variable(size=3,name="x")
    >>> mu_input = nnopinf.Variable(size=2,name="mu")
    >>> MatrixMlp = nnopinf.operators.MatrixOperator(n_outputs=5,acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
    """

    def __init__(self,n_outputs,acts_on,depends_on,n_hidden_layers,n_neurons_per_layer,activation=F.relu,name='MatrixOperator'):
        super(MatrixOperator, self).__init__()
        self.name_ = name
        self.n_outputs_ = n_outputs 
        self.acts_on_name_ = acts_on.get_name()
        self.depends_on_names_ = []
        self.n_inputs_ = 0
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())
          self.n_inputs_ += depends_on[i].get_size()

        network_output_size = acts_on.get_size() * self.n_outputs_ 
        self.acts_on_size_ = acts_on.get_size()
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        self.forward_list = create_layers(self.n_inputs_,network_output_size,n_hidden_layers,n_neurons_per_layer)
        self.activation = activation
        self.scaling_mat_ = torch.eye(self.n_outputs_)
        self.scaling_mat2_ = torch.eye(self.n_outputs_)

        self.scalings_set_ = False

    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool 
          If the method returns the (approximate) Jacobian, A. 
  
      Examples
      --------
      >>> import nnopinf
      >>> import nnopinf.operators
      >>> import numpy as np
      >>> x_input = nnopinf.Variable(size=3,name="x")
      >>> mu_input = nnopinf.Variable(size=2,name="mu")
      >>> MatrixMlp = nnopinf.operators.MatrixOperator(n_outputs = 5,acts_on=x_input,depends_on=(x_input,mu_input,),n_hidden_layers=2,n_neurons_per_layer=2)
      >>> inputs = {}
      >>> inputs['x'] = np.random.normal(3)
      >>> inputs['mu'] = np.random.normal(2)
      >>> Av,A = MatrixMlp.forward(inputs,True)
      """
      y = inputs_to_tensor(inputs,self.depends_on_names_) 
      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))
      y = self.forward_list[-1](y)
      A = torch.reshape(y,(y.shape[0],self.n_outputs_,self.acts_on_size_))
      A = torch.einsum('ij,njk->nik',self.scaling_mat2_,A)
      state = inputs[self.acts_on_name_] 
      state = torch.einsum('ij,nj->ni',self.scaling_mat_ , state)
      result = torch.einsum('ijk,ik->ij',A,state)
      if return_jacobian:
          return result[:,:],A
      else:
          return result[:,:]

    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      input_scalings = None
      for input_arg in self.depends_on_names_:
        if input_scalings is None:
          input_scalings = input_scalings_dict[input_arg]
        else:
          input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      
      # Update initial layer weights
      initial_layer = self.forward_list[0]
      initial_layer.weight[:] = initial_layer.weight[:] @ torch.eye(self.n_inputs_)/input_scalings
      # To do - make more efficient by editing weights
      #final_layer = self.forward_list[-1] 
      #final_layer.weight[:] =  ( torch.eye(self.n_outputs_) / input_scalings_dict[self.acts_on_name_] ) @ final_layer.weight[:] 
      #final_layer.bias[:] =  ( torch.eye(self.n_outputs_) / input_scalings_dict[self.acts_on_name_]).flatten() * final_layer.bias[:]
      self.scaling_mat_[:] = torch.eye(self.n_outputs_) *  output_scaling
      self.scaling_mat2_ = torch.eye(self.n_outputs_) /  input_scalings_dict[self.acts_on_name_]


class LinearAffineTensorOperator(nn.Module):
    """
    :math:`f: (v,x) \mapsto A_{ijk} x_k v_k`

    Constructs a linear affine tensor operator 

    Parameters
    ----------
    n_outputs: int
        Output dimension of the operator, i.e., ``M`` in the above description

    acts_on : nnopinf.Variable
        The state the operators acts on, i.e., the :math:`x`

    depends_on: tuple of nnopinf.Variable 
        The affine variables the operator depends on, i.e., the :math:`v`

    name : string 
        Operator name. Used when saving to file 

    """

    def __init__(self,n_outputs,acts_on,depends_on,name="LinearAffineTensorOperator"):
        super(LinearAffineTensorOperator, self).__init__()
        self.n_outputs_ = n_outputs 
        self.acts_on_name_ = acts_on.get_name()
        self.acts_on_size_ = acts_on.get_size()
        self.depends_on_names_ = []
        self.n_inputs_ = 0
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())
          self.n_inputs_ += depends_on[i].get_size()
        # Learnable tensor -- unstructured
        self.T = nn.Parameter(torch.randn([n_outputs,self.acts_on_size_,self.n_inputs_]))
        self.name_ = name


    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool 
          If the method returns the (approximate) Jacobian, A. 
  
      """

      # Separate out states and parameters
      x = inputs[self.acts_on_name_]
      mu = inputs_to_tensor(inputs,self.depends_on_names_)
      # Compute product (T*mu)x
      result = torch.einsum('ijp,bj,bp->bi',self.T,x,mu)

      if return_jacobian:
        jac = torch.einsum('ijp,bp->bij',self.T,mu)
        return result,jac
      else:
        return result


    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      input_scalings = None
      for input_arg in self.depends_on_names_:
        if input_scalings is None:
          input_scalings = input_scalings_dict[input_arg]
        else:
          input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      
      #for i in range(0,self.T.shape[-1]):
      self.input_scalings_ = input_scalings
      self.T[:,:,:] /= input_scalings[None,None,:]
      # Update initial layer weights
      state_scaling_mat = torch.eye(self.n_outputs_) / input_scalings_dict[self.acts_on_name_]
      self.T[:] = torch.einsum('ijk,jl->ilk',self.T[:],state_scaling_mat)
      output_scaling_mat = torch.eye(self.n_outputs_) * output_scaling 
      self.T[:] = torch.einsum('ij,jkl->ikl',output_scaling_mat,self.T[:])



class LinearAffineSkewTensorOperator(nn.Module):
    """
    :math:`f: (v,x) \mapsto [S_{ijk} - S_{jik}] x_k v_k`

    Constructs a linear affine skew-symmetric tensor operator 

    Parameters
    ----------
    acts_on : nnopinf.Variable
        The state the operators acts on, i.e., the :math:`x`

    depends_on: tuple of nnopinf.Variable 
        The affine variables the operator depends on, i.e., the :math:`v`


    skew : bool
        Constructs a skew-symmetric operator if true, constructs a symmetric operator if false 


    name : string 
        Operator name. Used when saving to file 

    """

    def __init__(self,acts_on,depends_on,skew=True,name="LinearAffineSkewTensorOperator"):
        super(LinearAffineSkewTensorOperator, self).__init__()
        self.name_ = name
        self.acts_on_name_ = acts_on.get_name()
        self.acts_on_size_ = acts_on.get_size()
        self.n_outputs_ = self.acts_on_size_ 

        self.depends_on_names_ = []
        self.n_inputs_ = 0
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())
          self.n_inputs_ += depends_on[i].get_size()

        # set dims and indices based on skew or sym
        if not skew: 
          dim = int(self.acts_on_size_*(self.acts_on_size_+1)/2)
          idx = torch.tril_indices(self.acts_on_size_,self.acts_on_size_,offset=0)
        else:
          dim = int(self.acts_on_size_*(self.acts_on_size_-1)/2)
          idx = torch.tril_indices(self.acts_on_size_,self.acts_on_size_,offset=-1)
        self.idx  = idx
        self.skew = skew

        # Learnable tensor -- stored as mtx
        self.mat = nn.Parameter(torch.randn([dim,self.n_inputs_]))

        # Scaling terms
        self.input_scalings_ = torch.ones(self.n_inputs_)
        self.state_scalings_ = torch.ones(self.n_outputs_)
        self.output_scalings_ = torch.ones(self.n_outputs_)

    def forward(self,inputs, return_jacobian=False):
      """
      Forward pass of operator 

      .. warning:: return_jacobian is currently not implemented 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool 
          If the method returns the (approximate) Jacobian, A. 
      """
      assert return_jacobian == False, "Return jacobian currently not implemented for linear skew operator"
      # Separate out states and parameters
      x = inputs[self.acts_on_name_]
      mu = inputs_to_tensor(inputs,self.depends_on_names_)
      mu = mu / self.input_scalings_
      # Fill tensor with learnable parameters
      S  = torch.zeros(self.n_outputs_,self.n_outputs_,self.n_inputs_)
      S[self.idx[0],self.idx[1],:] = self.mat

      # Make symmetric or skew
      St = torch.transpose(S,0,1) 
      T  = (S+St if not self.skew else S-St)

      # Compute product (T*mu)x
      x = x/self.state_scalings_
      result = torch.einsum('ijp,bj,bp->bi',T,x,mu)*self.output_scalings_
      return result[:,:]

    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      input_scalings = None
      for input_arg in self.depends_on_names_:
        if input_scalings is None:
          input_scalings = input_scalings_dict[input_arg]
        else:
          input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      # Update initial layer weights
      self.input_scalings_[:] = input_scalings
      self.output_scalings_[:] = output_scaling
      self.state_scalings_[:] = input_scalings_dict[self.acts_on_name_] 

class LinearAffineSpdTensorOperator(nn.Module):
    """
    :math:`f: (v,x) \mapsto \sum_{k=1}^{N} L_kD_kL_k^T x v_k`

    Constructs a linear affine SPD tensor operator 

    Parameters
    ----------
    acts_on : nnopinf.Variable
        The state the operators acts on, i.e., the :math:`x`

    depends_on: tuple of nnopinf.Variable 
        The affine variables the operator depends on, i.e., the :math:`v`

    positive : bool
        Constructs an SPD operator if true, and an NPD operator if flase 

    name : string 
        Operator name. Used when saving to file 

    """

    def __init__(self,acts_on,depends_on,positive=True,name="LinearAffineSpdTensorOperator"):
        super(LinearAffineSpdTensorOperator, self).__init__()

        self.name_ = name
        self.acts_on_name_ = acts_on.get_name()
        self.acts_on_size_ = acts_on.get_size()
        self.n_outputs_ = self.acts_on_size_ 

        self.depends_on_names_ = []
        self.n_inputs_ = 0
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())
          self.n_inputs_ += depends_on[i].get_size()

        # set dims and indices
        ldim      = int(self.n_outputs_*(self.n_outputs_-1)/2)
        self.lidx = torch.tril_indices(self.n_outputs_,self.n_outputs_,offset=-1)

        # Learnable tensor -- stored as 2 matrices
        # lmat is lower triangle and diag is diagonal part
        self.lmat = nn.Parameter(torch.randn([ldim,self.n_inputs_]))
        self.diag = nn.Parameter(torch.randn([self.n_outputs_,self.n_inputs_]))
        self.fac = (1 if positive else -1)

        # Scaling terms
        self.input_scalings_ = torch.ones(self.n_inputs_)
        self.state_scalings_ = torch.ones(self.n_outputs_)
        self.output_scalings_ = torch.ones(self.n_outputs_)


    def forward(self,inputs):
      """
      Forward pass of operator 

      .. warning:: return_jacobian is currently not implemented 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool 
          If the method returns the (approximate) Jacobian, A. 
      """
      assert return_jacobian == False, "Return jacobian currently not implemented for linear skew operator"

      # Separate out states and parameters
      x = inputs[self.acts_on_name_]
      mu = inputs_to_tensor(inputs,self.depends_on_names_)
      mu = mu / self.input_scalings_

      # Fill tensor with learnable parameters
      # Softplus hits diagonal, lower triangle is whatever
      S = torch.zeros(self.n_outputs_,self.n_outputs_,self.n_inputs_)
      S[self.lidx[0],self.lidx[1],:] = self.lmat
      S[range(self.n_outputs_),range(self.n_outputs_),:] = F.softplus(self.diag)

      # Symmetrize
      St = torch.transpose(S,0,1) 
      T  = self.fac * (S + St)

      # Compute product (T*mu)x
      x = x/self.state_scalings_
      result = torch.einsum('ijp,bj,bp->bi',T,x,mu)*self.output_scalings_
      return result[:,:]

    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      input_scalings = None
      for input_arg in self.depends_on_names_:
        if input_scalings is None:
          input_scalings = input_scalings_dict[input_arg]
        else:
          input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      # Update initial layer weights
      self.input_scalings_[:] = input_scalings
      self.output_scalings_[:] = output_scaling
      self.state_scalings_[:] = input_scalings_dict[self.acts_on_name_] 



if __name__ == "__main__":
    n_hidden_layers = 2
    n_neurons_per_layer = 5
    x = nnopinf.Variable(size=5,name='x')
    mu = nnopinf.Variable(size=3,name='mu')

    StandardMlp = StandardOperator(n_outputs=7,depends_on=[x,mu],n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer)
    SpdMlp = SpdOperator(acts_on=x,depends_on=[x,mu],n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer,positive=True)
    SkewMlp = SkewOperator(acts_on=x,depends_on=[x,mu],n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer)
    CompositeMlp = CompositeOperator([SpdMlp,SkewMlp])
    MatrixMlp = MatrixOperator(n_outputs=7,acts_on=x,depends_on=[x,mu],n_hidden_layers=n_hidden_layers,n_neurons_per_layer=n_neurons_per_layer)
    LinearAffineMlp = LinearAffineTensorOperator(n_outputs=7,acts_on=x,depends_on=[mu])
    inputs = {'x':torch.ones((1,5)),'mu':torch.ones((1,3))}
    r1 = StandardMlp.forward(inputs)
    r2 = SpdMlp.forward(inputs)
    r3 = SkewMlp.forward(inputs)
    r4 = MatrixMlp.forward(inputs)
    r5 = CompositeMlp.forward(inputs)
    r6 = LinearAffineMlp.forward(inputs)

    assert np.allclose( (r2+r3).detach().numpy(),r5.detach().numpy())
