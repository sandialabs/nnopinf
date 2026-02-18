import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import nnopinf.variables
from typing import Protocol

torch.set_default_dtype(torch.float64)

def siren_activation(x):
    return torch.sin(x)


def _is_siren_activation(activation):
    return activation is torch.sin or activation is siren_activation


def _apply_activation(activation, z, omega_0):
    if _is_siren_activation(activation):
        return activation(omega_0 * z)
    return activation(z)


def _init_siren_layers(layers, omega_0):
    for i, layer in enumerate(layers):
        in_dim = layer.weight.shape[1]
        if i == 0:
            bound = 1.0 / in_dim
        else:
            bound = np.sqrt(6.0 / in_dim) / omega_0
        nn.init.uniform_(layer.weight, -bound, bound)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -bound, bound)


def _siren_activation_for_layer(activation, omega_0, layer_index):
    if _is_siren_activation(activation):
        if layer_index == 0:
            return lambda z: activation(omega_0 * z)
        return activation
    return activation


def _init_siren_first_layer(layer, omega_0):
    in_dim = layer.weight.shape[1]
    bound = 1.0 / in_dim
    nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -bound, bound)


def _augment_first_layer_with_siren(activation, omega_0, z):
    return activation(z) + torch.sin(omega_0 * z)


def _augment_tensor_with_fourier(x, freqs):
    if freqs is None or freqs.numel() == 0:
        return x
    freqs = freqs.to(device=x.device, dtype=x.dtype)
    x_exp = x.unsqueeze(-1) * freqs.view(1, 1, -1)
    sin = torch.sin(x_exp).reshape(x.shape[0], -1)
    cos = torch.cos(x_exp).reshape(x.shape[0], -1)
    return torch.cat([x, sin, cos], dim=1)

def inputs_to_tensor(inputs,names_to_collect):
    operator_inputs = []
    for input_name in names_to_collect:
      operator_inputs.append(inputs[input_name])
    if len(operator_inputs) == 0:
      return torch.zeros((0,0))
    return torch.cat(operator_inputs, dim=1)

def _fourier_frequencies(fourier_frequencies, num_frequencies, base, scale, device, dtype):
    if fourier_frequencies is not None:
        freqs = torch.as_tensor(fourier_frequencies, dtype=dtype, device=device)
        return freqs
    if num_frequencies <= 0:
        return torch.zeros((0,), dtype=dtype, device=device)
    freqs = scale * (base ** torch.arange(num_frequencies, dtype=dtype, device=device))
    return freqs


def _augmented_input_size(depends_on, fourier_variables, num_frequencies):
    if not fourier_variables or num_frequencies <= 0:
        return sum(var.get_size() for var in depends_on)
    total = 0
    for var in depends_on:
        size = var.get_size()
        total += size
        if var.get_name() in fourier_variables:
            total += size * 2 * num_frequencies
    return total


def _inputs_to_tensor_with_features(
    inputs,
    names_to_collect,
    fourier_variables,
    fourier_frequencies,
    input_scalings_dict=None,
):
    operator_inputs = []
    for input_name in names_to_collect:
        x = inputs[input_name]
        if input_scalings_dict is not None and input_name in input_scalings_dict:
            x = x / input_scalings_dict[input_name]
        if fourier_variables is not None and input_name in fourier_variables:
            freqs = fourier_frequencies
            if freqs is not None and freqs.numel() > 0:
                freqs = freqs.to(device=x.device, dtype=x.dtype)
                x_exp = x.unsqueeze(-1) * freqs.view(1, 1, -1)
                sin = torch.sin(x_exp).reshape(x.shape[0], -1)
                cos = torch.cos(x_exp).reshape(x.shape[0], -1)
                x = torch.cat([x, sin, cos], dim=1)
        operator_inputs.append(x)
    if len(operator_inputs) == 0:
        return torch.zeros((0, 0))
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

    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 

      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
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

    def forward(self,inputs,return_jacobian=False):
      """ 
      Forward pass of operator 

      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
  
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

    siren_omega_0 : float
        Frequency scale for SIREN activations.

    siren_first_layer : bool
        If True, use a SIREN-style first layer even for non-sine activations.

    fourier_features : bool
        If True, augment inputs with Fourier features.

    fourier_variables : iterable of str, optional
        Variable names to augment with Fourier features. Defaults to all inputs.

    fourier_num_frequencies : int
        Number of frequencies per variable when using Fourier features.

    fourier_base : float
        Base for geometric progression of Fourier frequencies.

    fourier_scale : float
        Scaling applied to Fourier frequencies.

    fourier_frequencies : array-like, optional
        Explicit list of Fourier frequencies. Overrides ``fourier_num_frequencies``.
 
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

    def __init__(self,n_outputs,depends_on,n_hidden_layers,n_neurons_per_layer,activation=torch.tanh,siren_omega_0=30.0,siren_first_layer=False,fourier_features=False,fourier_variables=None,fourier_num_frequencies=2,fourier_base=2.0,fourier_scale=1.0,fourier_frequencies=None,name='StandardOperator'):
        super(StandardOperator, self).__init__()
        self.name_ = name
        self.n_outputs_ = n_outputs 
        self.depends_on_names_ = []
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())

        network_output_size = self.n_outputs_ 
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        freq_count = len(fourier_frequencies) if fourier_frequencies is not None else fourier_num_frequencies
        n_inputs = _augmented_input_size(depends_on, set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_), freq_count) if fourier_features else sum(var.get_size() for var in depends_on)
        self.n_inputs_ = n_inputs
        self.forward_list = create_layers(self.n_inputs_,network_output_size,n_hidden_layers,n_neurons_per_layer)
        if _is_siren_activation(activation):
          _init_siren_layers(self.forward_list, siren_omega_0)
        elif siren_first_layer and len(self.forward_list) > 0:
          _init_siren_first_layer(self.forward_list[0], siren_omega_0)
        self.activation = activation
        self.activation_omega_0 = siren_omega_0
        self.siren_first_layer_ = siren_first_layer
        self.fourier_features_ = fourier_features
        self.fourier_variables_ = set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_)
        self.fourier_frequencies_ = _fourier_frequencies(
            fourier_frequencies,
            fourier_num_frequencies,
            fourier_base,
            fourier_scale,
            device=torch.device('cpu'),
            dtype=torch.float64,
        )
        self.input_scalings_dict_ = {}
        self.scalings_set_ = False
        self.input_scaling_ = torch.ones(n_inputs)
        self.output_scaling_ = torch.ones(n_outputs)

    def _net(self,y):
      for i in range(0,self.num_layers-1):
        if self.siren_first_layer_ and i == 0 and not _is_siren_activation(self.activation):
          y = _augment_first_layer_with_siren(
              self.activation, self.activation_omega_0, self.forward_list[i](y)
          )
        else:
          act = _siren_activation_for_layer(self.activation, self.activation_omega_0, i)
          y = act(self.forward_list[i](y))
      return self.forward_list[-1](y)

    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
  
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
      >>> f = StandardMlp.forward(inputs)
      """

      if self.fourier_features_:
        y = _inputs_to_tensor_with_features(
            inputs,
            self.depends_on_names_,
            self.fourier_variables_,
            self.fourier_frequencies_,
            self.input_scalings_dict_ if self.scalings_set_ else None,
        )
      else:
        y = inputs_to_tensor(inputs,self.depends_on_names_)
      result = self._net(y)
      if return_jacobian:
        jac_list = []
        for i in range(result.shape[0]):
          yi = y[i].detach().requires_grad_(True)

          def scalar_fn(z):
            return self._net(z.unsqueeze(0)).squeeze(0)

          jac = torch.autograd.functional.jacobian(scalar_fn, yi, create_graph=True)
          jac_list.append(jac)
        jacobian = torch.stack(jac_list, dim=0)
        return result[:,:], jacobian
      else:
        return result[:,:]


    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      if self.fourier_features_:
        self.input_scalings_dict_ = input_scalings_dict
      input_scalings = None
      if not self.fourier_features_:
        for input_arg in self.depends_on_names_:
          if input_scalings is None:
            input_scalings = input_scalings_dict[input_arg]
          else:
            input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      
      # Update initial layer weights
      if not self.fourier_features_:
        initial_layer = self.forward_list[0]
        initial_layer.weight[:] = initial_layer.weight[:] @ torch.eye(self.n_inputs_)/input_scalings
       
      # Update final layer weights
      final_layer = self.forward_list[-1]
      final_layer.weight[:] =  (output_scaling * torch.eye(self.n_outputs_)) @ final_layer.weight[:]
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
 
    siren_omega_0 : float
        Frequency scale for SIREN activations.

    siren_first_layer : bool
        If True, use a SIREN-style first layer even for non-sine activations.

    fourier_features : bool
        If True, augment inputs with Fourier features.

    fourier_variables : iterable of str, optional
        Variable names to augment with Fourier features. Defaults to all inputs.

    fourier_num_frequencies : int
        Number of frequencies per variable when using Fourier features.

    fourier_base : float
        Base for geometric progression of Fourier frequencies.

    fourier_scale : float
        Scaling applied to Fourier frequencies.

    fourier_frequencies : array-like, optional
        Explicit list of Fourier frequencies. Overrides ``fourier_num_frequencies``.

    positive : bool 
        If operator is SPD or NPD 

    parameterization : str
        SPD parameterization. Supported values are ``"cholesky"`` (default, uses
        :math:`L L^T`) and ``"matrix_exp"`` (uses :math:`\\exp(S)` with symmetric
        :math:`S`).

    residual : bool
        If True, use residual connections between hidden layers when shapes match.

    layer_norm : bool
        If True, apply LayerNorm after each hidden linear layer.

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

    def __init__(
        self,
        acts_on,
        depends_on,
        n_hidden_layers,
        n_neurons_per_layer,
        activation=torch.tanh,
        siren_omega_0=30.0,
        siren_first_layer=False,
        fourier_features=False,
        fourier_variables=None,
        fourier_num_frequencies=6,
        fourier_base=2.0,
        fourier_scale=1.0,
        fourier_frequencies=None,
        positive=True,
        name='SpdOperator',
        parameterization='cholesky',
        residual=True,
        layer_norm=True,
    ):
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
        if parameterization not in ('cholesky', 'matrix_exp'):
          raise ValueError("parameterization must be 'cholesky' or 'matrix_exp'")
        self.parameterization_ = parameterization
        self.residual_ = residual
        self.layer_norm_ = layer_norm
        self.activation_omega_0 = siren_omega_0
        self.siren_first_layer_ = siren_first_layer
        self.fourier_features_ = fourier_features
        self.fourier_variables_ = set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_)
        self.fourier_frequencies_ = _fourier_frequencies(
            fourier_frequencies,
            fourier_num_frequencies,
            fourier_base,
            fourier_scale,
            device=torch.device('cpu'),
            dtype=torch.float64,
        )
        self.input_scalings_dict_ = {}
        self.scalings_set_ = False
        self.siren_first_layer_ = siren_first_layer

        network_output_size = idx[0].size
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        freq_count = len(fourier_frequencies) if fourier_frequencies is not None else fourier_num_frequencies
        self.n_inputs_ = _augmented_input_size(
            depends_on,
            set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_),
            freq_count,
        ) if fourier_features else sum(var.get_size() for var in depends_on)

        if self.positive_:
          self.scale_ = 1.0
        else:
          self.scale_ = -1.0
        self.forward_list = create_layers(self.n_inputs_,network_output_size,n_hidden_layers,n_neurons_per_layer)
        if _is_siren_activation(activation):
          _init_siren_layers(self.forward_list, siren_omega_0)
        elif siren_first_layer and len(self.forward_list) > 0:
          _init_siren_first_layer(self.forward_list[0], siren_omega_0)
        if self.layer_norm_:
          self.layer_norms_ = nn.ModuleList(
            [nn.LayerNorm(n_neurons_per_layer) for _ in range(self.num_layers-1)]
          )
        self.activation = activation
        self.scaling_mat_ = torch.eye(self.n_outputs_)

    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
  
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

      if self.fourier_features_:
        y = _inputs_to_tensor_with_features(
            inputs,
            self.depends_on_names_,
            self.fourier_variables_,
            self.fourier_frequencies_,
            self.input_scalings_dict_ if self.scalings_set_ else None,
        )
      else:
        y = inputs_to_tensor(inputs,self.depends_on_names_)
      for i in range(0,self.num_layers-1):
        z = self.forward_list[i](y)
        if self.layer_norm_:
          z = self.layer_norms_[i](z)
        if self.siren_first_layer_ and i == 0 and not _is_siren_activation(self.activation):
          z = _augment_first_layer_with_siren(
              self.activation, self.activation_omega_0, z
          )
        else:
          act = _siren_activation_for_layer(self.activation, self.activation_omega_0, i)
          z = act(z)
        if self.residual_ and z.shape == y.shape:
          y = y + 0.05*z
        else:
          y = z

      y = self.forward_list[-1](y)
      K = torch.zeros(y.shape[0],self.n_outputs_,self.n_outputs_)
      K[:,self.idx[0],self.idx[1]] = y[:,0:self.idx[0].size]
      if self.parameterization_ == 'matrix_exp':
        KT = torch.transpose(K,2,1)
        K = K + KT - torch.diag_embed(torch.diagonal(K, dim1=1, dim2=2))
        K = torch.matrix_exp(K)
      else:
        KT = torch.transpose(K,2,1)
        K = torch.einsum('ijk,ikl->ijl',K,KT)
      state = inputs[self.acts_on_name_] 
      state = torch.einsum('ij,nj->ni',self.scaling_mat_ , state)
      result = torch.einsum('ijk,ik->ij',K,state )
      
      if return_jacobian:
          jac = torch.einsum('ijk,kl->ijl',K,self.scaling_mat_)
          return result*self.scale_, jac*self.scale_
      else:
          return result*self.scale_

    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      if self.fourier_features_:
        self.input_scalings_dict_ = input_scalings_dict
      input_scalings = None
      if not self.fourier_features_:
        for input_arg in self.depends_on_names_:
          if input_scalings is None:
            input_scalings = input_scalings_dict[input_arg]
          else:
            input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      
      # Update initial layer weights
      if not self.fourier_features_:
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

    siren_omega_0 : float
        Frequency scale for SIREN activations.

    siren_first_layer : bool
        If True, use a SIREN-style first layer even for non-sine activations.

    fourier_features : bool
        If True, augment inputs with Fourier features.

    fourier_variables : iterable of str, optional
        Variable names to augment with Fourier features. Defaults to all inputs.

    fourier_num_frequencies : int
        Number of frequencies per variable when using Fourier features.

    fourier_base : float
        Base for geometric progression of Fourier frequencies.

    fourier_scale : float
        Scaling applied to Fourier frequencies.

    fourier_frequencies : array-like, optional
        Explicit list of Fourier frequencies. Overrides ``fourier_num_frequencies``.
 
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

    def __init__(self,acts_on,depends_on,n_hidden_layers,n_neurons_per_layer,activation=torch.tanh, siren_omega_0=30.0, siren_first_layer=False, fourier_features=False, fourier_variables=None, fourier_num_frequencies=2, fourier_base=2.0, fourier_scale=1.0, fourier_frequencies=None, name='SkewOperator'):
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

        freq_count = len(fourier_frequencies) if fourier_frequencies is not None else fourier_num_frequencies
        self.n_inputs_ = _augmented_input_size(
            depends_on,
            set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_),
            freq_count,
        ) if fourier_features else sum(var.get_size() for var in depends_on)
        self.forward_list = create_layers(self.n_inputs_,network_output_size,n_hidden_layers,n_neurons_per_layer)
        if _is_siren_activation(activation):
          _init_siren_layers(self.forward_list, siren_omega_0)
        elif siren_first_layer and len(self.forward_list) > 0:
          _init_siren_first_layer(self.forward_list[0], siren_omega_0)

        self.activation = activation
        self.activation_omega_0 = siren_omega_0
        self.siren_first_layer_ = siren_first_layer
        self.fourier_features_ = fourier_features
        self.fourier_variables_ = set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_)
        self.fourier_frequencies_ = _fourier_frequencies(
            fourier_frequencies,
            fourier_num_frequencies,
            fourier_base,
            fourier_scale,
            device=torch.device('cpu'),
            dtype=torch.float64,
        )
        self.input_scalings_dict_ = {}
        self.scalings_set_ = False
        self.scaling_mat_ = torch.eye(self.n_outputs_)     


    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
  
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

      if self.fourier_features_:
        y = _inputs_to_tensor_with_features(
            inputs,
            self.depends_on_names_,
            self.fourier_variables_,
            self.fourier_frequencies_,
            self.input_scalings_dict_ if self.scalings_set_ else None,
        )
      else:
        y = inputs_to_tensor(inputs,self.depends_on_names_)
      for i in range(0,self.num_layers-1):
        if self.siren_first_layer_ and i == 0 and not _is_siren_activation(self.activation):
          y = _augment_first_layer_with_siren(
              self.activation, self.activation_omega_0, self.forward_list[i](y)
          )
        else:
          act = _siren_activation_for_layer(self.activation, self.activation_omega_0, i)
          y = act(self.forward_list[i](y))

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
      if self.fourier_features_:
        self.input_scalings_dict_ = input_scalings_dict
      input_scalings = None
      if not self.fourier_features_:
        for input_arg in self.depends_on_names_:
          if input_scalings is None:
            input_scalings = input_scalings_dict[input_arg]
          else:
            input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      
      # Update initial layer weights
      if not self.fourier_features_:
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

    siren_omega_0 : float
        Frequency scale for SIREN activations.

    siren_first_layer : bool
        If True, use a SIREN-style first layer even for non-sine activations.

    fourier_features : bool
        If True, augment inputs with Fourier features.

    fourier_variables : iterable of str, optional
        Variable names to augment with Fourier features. Defaults to all inputs.

    fourier_num_frequencies : int
        Number of frequencies per variable when using Fourier features.

    fourier_base : float
        Base for geometric progression of Fourier frequencies.

    fourier_scale : float
        Scaling applied to Fourier frequencies.

    fourier_frequencies : array-like, optional
        Explicit list of Fourier frequencies. Overrides ``fourier_num_frequencies``.
 
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

    def __init__(self,n_outputs,acts_on,depends_on,n_hidden_layers,n_neurons_per_layer,activation=torch.tanh, siren_omega_0=30.0, siren_first_layer=False, fourier_features=False, fourier_variables=None, fourier_num_frequencies=2, fourier_base=2.0, fourier_scale=1.0, fourier_frequencies=None, name='MatrixOperator'):
        super(MatrixOperator, self).__init__()
        self.name_ = name
        self.n_outputs_ = n_outputs 
        self.acts_on_name_ = acts_on.get_name()
        self.depends_on_names_ = []
        self.n_inputs_ = 0
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())
          self.n_inputs_ += depends_on[i].get_size()
        freq_count = len(fourier_frequencies) if fourier_frequencies is not None else fourier_num_frequencies
        if fourier_features:
          self.n_inputs_ = _augmented_input_size(
              depends_on,
              set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_),
              freq_count,
          )
        network_output_size = acts_on.get_size() * self.n_outputs_ 
        self.acts_on_size_ = acts_on.get_size()
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        self.forward_list = create_layers(self.n_inputs_,network_output_size,n_hidden_layers,n_neurons_per_layer)
        if _is_siren_activation(activation):
          _init_siren_layers(self.forward_list, siren_omega_0)
        elif siren_first_layer and len(self.forward_list) > 0:
          _init_siren_first_layer(self.forward_list[0], siren_omega_0)
        self.activation = activation
        self.activation_omega_0 = siren_omega_0
        self.siren_first_layer_ = siren_first_layer
        self.fourier_features_ = fourier_features
        self.fourier_variables_ = set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_)
        self.fourier_frequencies_ = _fourier_frequencies(
            fourier_frequencies,
            fourier_num_frequencies,
            fourier_base,
            fourier_scale,
            device=torch.device('cpu'),
            dtype=torch.float64,
        )
        self.input_scalings_dict_ = {}
        self.scalings_set_ = False
        self.scaling_mat_ = torch.eye(self.n_outputs_)
        self.scaling_mat2_ = torch.eye(self.n_outputs_)
        self.scaling_inputs_ = np.ones(self.acts_on_size_) 

        self.scalings_set_ = False

    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
  
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
      if self.fourier_features_:
        y = _inputs_to_tensor_with_features(
            inputs,
            self.depends_on_names_,
            self.fourier_variables_,
            self.fourier_frequencies_,
            self.input_scalings_dict_ if self.scalings_set_ else None,
        )
      else:
        y = inputs_to_tensor(inputs,self.depends_on_names_) 
      for i in range(0,self.num_layers-1):
        if self.siren_first_layer_ and i == 0 and not _is_siren_activation(self.activation):
          y = _augment_first_layer_with_siren(
              self.activation, self.activation_omega_0, self.forward_list[i](y)
          )
        else:
          act = _siren_activation_for_layer(self.activation, self.activation_omega_0, i)
          y = act(self.forward_list[i](y))
      y = self.forward_list[-1](y)
      A = torch.reshape(y,(y.shape[0],self.n_outputs_,self.acts_on_size_))
      A = torch.einsum('ij,njk->nik',self.scaling_mat_,A)
      state = inputs[self.acts_on_name_] 
      #print(self.scaling_mat_.shape,state.shape)
      state = state * self.scaling_inputs_[None] 
      #state = torch.einsum('ij,nj->ni',self.scaling_mat_ , state)
      result = torch.einsum('ijk,ik->ij',A,state)
      if return_jacobian:
          return result[:,:],A
      else:
          return result[:,:]

    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      if self.fourier_features_:
        self.input_scalings_dict_ = input_scalings_dict
      input_scalings = None
      if not self.fourier_features_:
        for input_arg in self.depends_on_names_:
          if input_scalings is None:
            input_scalings = input_scalings_dict[input_arg]
          else:
            input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      
      # Update initial layer weights
      if not self.fourier_features_:
        initial_layer = self.forward_list[0]
        initial_layer.weight[:] = initial_layer.weight[:] @ torch.eye(self.n_inputs_)/input_scalings
      # To do - make more efficient by editing weights
      #final_layer = self.forward_list[-1] 
      #final_layer.weight[:] =  ( torch.eye(self.n_outputs_) / input_scalings_dict[self.acts_on_name_] ) @ final_layer.weight[:] 
      #final_layer.bias[:] =  ( torch.eye(self.n_outputs_) / input_scalings_dict[self.acts_on_name_]).flatten() * final_layer.bias[:]
      self.scaling_mat_[:] = torch.eye(self.n_outputs_) *  output_scaling
      #self.scaling_mat2_ = torch.eye(self.n_outputs_) /  input_scalings_dict[self.acts_on_name_]
      self.scaling_inputs_ = 1./ input_scalings_dict[self.acts_on_name_]


class QuadraticOperator(nn.Module):
    """
    :math:`f: x \mapsto H (x \\otimes x)`

    Constructs a quadratic operator with a learnable matrix :math:`H` that acts
    on the Kronecker product of the state with itself.

    Parameters
    ----------
    n_outputs: int
        Output dimension of the operator.

    acts_on : nnopinf.Variable
        The state the operator acts on, i.e., the :math:`x` in
        :math:`H (x \\otimes x)`.

    name : string
        Operator name. Used when saving to file.
    """

    def __init__(self,n_outputs,acts_on,name="QuadraticOperator"):
        super(QuadraticOperator, self).__init__()
        self.name_ = name
        self.n_outputs_ = n_outputs
        self.acts_on_name_ = acts_on.get_name()
        self.acts_on_size_ = acts_on.get_size()
        self.H = nn.Parameter(torch.randn(n_outputs, self.acts_on_size_ * self.acts_on_size_))
        self.scalings_set_ = False

    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator

      Parameters
      ----------
      inputs : dict(str, np.array)
          Dictionary of input data in the form of arrays referenced by the
          variable name, i.e., inputs['x'] = np.ones(3)

      return_jacobian: bool, optional
          If True, return the Jacobian with respect to the state.
      """
      x = inputs[self.acts_on_name_]
      x_outer = torch.einsum('bi,bj->bij',x,x)
      x_kron = x_outer.reshape(x.shape[0],-1)
      result = torch.einsum('ij,bj->bi',self.H,x_kron)
      if return_jacobian:
          H_mat = self.H.view(self.n_outputs_,self.acts_on_size_,self.acts_on_size_)
          term1 = torch.einsum('mij,bj->bmi',H_mat,x)
          term2 = torch.einsum('mij,bi->bmj',H_mat,x)
          jac = term1 + term2
          return result,jac
      else:
          return result

    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      input_scalings = input_scalings_dict[self.acts_on_name_]
      kron_scalings = torch.einsum('i,j->ij',input_scalings,input_scalings).reshape(-1)
      self.H[:] = (output_scaling[:,None] * self.H) / kron_scalings[None,:]


class StandardLagrangianOperator(nn.Module):
    """
    :math:`f: x \\mapsto \\nabla_x \\mathcal{L}(x)`

    Constructs a Lagrangian operator that returns the gradient of a scalar
    network with respect to the state.

    Parameters
    ----------
    n_outputs: int
        Output dimension of the operator.

    depends_on: tuple of nnopinf.Variable
        The variables the operator depends on.

    n_hidden_layers : int
        Number of hidden layers in the network.

    n_neurons_per_layer : int
        Number of neurons in each hidden layer.

    activation : PyTorch activation function (e.g., torch.tanh)
        Activation function used at each layer.

    siren_omega_0 : float
        Frequency scale for SIREN activations.

    siren_first_layer : bool
        If True, use a SIREN-style first layer even for non-sine activations.

    fourier_features : bool
        If True, augment inputs with Fourier features.

    fourier_variables : iterable of str, optional
        Variable names to augment with Fourier features. Defaults to all inputs.

    fourier_num_frequencies : int
        Number of frequencies per variable when using Fourier features.

    fourier_base : float
        Base for geometric progression of Fourier frequencies.

    fourier_scale : float
        Scaling applied to Fourier frequencies.

    fourier_frequencies : array-like, optional
        Explicit list of Fourier frequencies. Overrides ``fourier_num_frequencies``.

    name : string
        Operator name. Used when saving to file.
    """

    def __init__(self,n_outputs,depends_on,n_hidden_layers,n_neurons_per_layer,activation=torch.tanh, siren_omega_0=30.0, siren_first_layer=False, fourier_features=False, fourier_variables=None, fourier_num_frequencies=2, fourier_base=2.0, fourier_scale=1.0, fourier_frequencies=None, name="StandardLagrangianOperator"):
        super(StandardLagrangianOperator, self).__init__()
        self.name_ = name
        self.depends_on_names_ = []
        self.input_size_ = 0
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())
          self.input_size_ += depends_on[i].get_size()

        assert len(depends_on) == 1, "StandardLagrangianOperator supports a single state variable"
        assert n_outputs == self.input_size_, "n_outputs must match the input size"

        self.n_outputs_ = n_outputs
        freq_count = len(fourier_frequencies) if fourier_frequencies is not None else fourier_num_frequencies
        if fourier_features:
          self.net_input_size_ = _augmented_input_size(
              depends_on,
              set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_),
              freq_count,
          )
        else:
          self.net_input_size_ = self.input_size_
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        layer_dims = [self.net_input_size_] + [n_neurons_per_layer] * n_hidden_layers + [1]
        forward_list = []
        for i in range(1, len(layer_dims)):
          in_dim = layer_dims[i - 1]
          out_dim = layer_dims[i]
          bias = False if i == len(layer_dims) - 1 else True
          forward_list.append(nn.Linear(in_dim, out_dim, bias=bias))
        self.forward_list = nn.ModuleList(forward_list)
        if _is_siren_activation(activation):
          _init_siren_layers(self.forward_list, siren_omega_0)
        elif siren_first_layer and len(self.forward_list) > 0:
          _init_siren_first_layer(self.forward_list[0], siren_omega_0)
        self.activation = activation
        self.activation_omega_0 = siren_omega_0
        self.siren_first_layer_ = siren_first_layer
        self.fourier_features_ = fourier_features
        self.fourier_variables_ = set(fourier_variables) if fourier_variables is not None else set(self.depends_on_names_)
        self.fourier_frequencies_ = _fourier_frequencies(
            fourier_frequencies,
            fourier_num_frequencies,
            fourier_base,
            fourier_scale,
            device=torch.device('cpu'),
            dtype=torch.float64,
        )
        self.input_scalings_dict_ = {}

        self.input_scalings_ = torch.ones(self.input_size_)
        self.output_scalings_ = torch.ones(self.n_outputs_)
        self.scalings_set_ = False

    def _build_features(self, x):
        if self.fourier_features_ and self.depends_on_names_[0] in self.fourier_variables_:
            return _augment_tensor_with_fourier(x, self.fourier_frequencies_)
        return x

    def _net(self,x_feat):
        y = x_feat
        for i in range(0,self.num_layers-1):
          if self.siren_first_layer_ and i == 0 and not _is_siren_activation(self.activation):
            y = _augment_first_layer_with_siren(
                self.activation, self.activation_omega_0, self.forward_list[i](y)
            )
          else:
            act = _siren_activation_for_layer(self.activation, self.activation_omega_0, i)
            y = act(self.forward_list[i](y))
        return self.forward_list[-1](y)

    def _grad(self,x):
        x = x.detach().requires_grad_(True)
        x_feat = self._build_features(x)
        y = torch.sum(self._net(x_feat))
        grad, = torch.autograd.grad(y, x, create_graph=True)
        return grad

    def _hessian(self,x):
        hessians = []
        for i in range(x.shape[0]):
          xi = x[i]
          def scalar_fn(z):
            z_feat = self._build_features(z.unsqueeze(0))
            return torch.sum(self._net(z_feat))
          hess = torch.autograd.functional.hessian(scalar_fn, xi, create_graph=True)
          hessians.append(hess)
        return torch.stack(hessians, dim=0)

    def forward(self,inputs,return_jacobian=False):
      """
      Forward pass of operator

      Parameters
      ----------
      inputs : dict(str, np.array)
          Dictionary of input data in the form of arrays referenced by the
          variable name, i.e., inputs['x'] = np.ones(3)

      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
      """
      x = inputs_to_tensor(inputs,self.depends_on_names_)
      x_scaled = x / self.input_scalings_
      grad_scaled = self._grad(x_scaled)
      grad = (grad_scaled * self.output_scalings_) / self.input_scalings_
      if return_jacobian:
          hess_scaled = self._hessian(x_scaled)
          row_scale = (self.output_scalings_ / self.input_scalings_).unsqueeze(0).unsqueeze(2)
          col_scale = (1.0 / self.input_scalings_).unsqueeze(0).unsqueeze(1)
          jac = row_scale * hess_scaled * col_scale
          return -grad[:,:], -jac
      else:
          return -grad[:,:]

    def set_scalings(self,input_scalings_dict,output_scaling):
     with torch.no_grad():
      self.scalings_set_ = True
      if self.fourier_features_:
        self.input_scalings_dict_ = input_scalings_dict
      input_scalings = None
      for input_arg in self.depends_on_names_:
        if input_scalings is None:
          input_scalings = input_scalings_dict[input_arg]
        else:
          input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      self.input_scalings_[:] = input_scalings
      self.output_scalings_[:] = output_scaling


class PsdLagrangianOperator(nn.Module):
    """
    :math:`f: x \\mapsto \\nabla_x \\mathcal{L}(x)` where
    :math:`\\mathcal{L}(x) = x^T A(x) x` and :math:`A(x)` is an SPD operator.

    Constructs a Lagrangian operator whose internal scalar network is formed
    by an SPD operator acting on the state and then left-multiplied by the state.

    Parameters
    ----------
    acts_on : nnopinf.Variable
        The state the operator acts on, i.e., the :math:`x`.

    depends_on: tuple of nnopinf.Variable
        The variables the operator depends on.

    n_hidden_layers : int
        Number of hidden layers in the SPD network.

    n_neurons_per_layer : int
        Number of neurons in each hidden layer.

    activation : PyTorch activation function (e.g., torch.nn.functional.relu)
        Activation function used at each layer.

    siren_omega_0 : float
        Frequency scale for SIREN activations.

    siren_first_layer : bool
        If True, use a SIREN-style first layer even for non-sine activations.

    fourier_features : bool
        If True, augment inputs with Fourier features.

    fourier_variables : iterable of str, optional
        Variable names to augment with Fourier features. Defaults to all inputs.

    fourier_num_frequencies : int
        Number of frequencies per variable when using Fourier features.

    fourier_base : float
        Base for geometric progression of Fourier frequencies.

    fourier_scale : float
        Scaling applied to Fourier frequencies.

    fourier_frequencies : array-like, optional
        Explicit list of Fourier frequencies. Overrides ``fourier_num_frequencies``.

    positive : bool
        If operator is SPD or NPD.

    name : string
        Operator name. Used when saving to file.
    """

    def __init__(
        self,
        acts_on,
        depends_on,
        n_hidden_layers,
        n_neurons_per_layer,
        activation=torch.tanh,
        siren_omega_0=30.0,
        siren_first_layer=False,
        fourier_features=False,
        fourier_variables=None,
        fourier_num_frequencies=2,
        fourier_base=2.0,
        fourier_scale=1.0,
        fourier_frequencies=None,
        positive=True,
        name="PsdLagrangianOperator",
    ):
        super(PsdLagrangianOperator, self).__init__()
        self.name_ = name
        self.acts_on_name_ = acts_on.get_name()
        self.acts_on_size_ = acts_on.get_size()
        self.depends_on_names_ = []
        self.n_inputs_ = 0
        for i in range(len(depends_on)):
            self.depends_on_names_.append(depends_on[i].get_name())
            self.n_inputs_ += depends_on[i].get_size()

        self.n_outputs_ = self.acts_on_size_
        self.spd_operator_ = SpdOperator(
            acts_on=acts_on,
            depends_on=depends_on,
            n_hidden_layers=n_hidden_layers,
            n_neurons_per_layer=n_neurons_per_layer,
            activation=activation,
            siren_omega_0=siren_omega_0,
            siren_first_layer=siren_first_layer,
            fourier_features=fourier_features,
            fourier_variables=fourier_variables,
            fourier_num_frequencies=fourier_num_frequencies,
            fourier_base=fourier_base,
            fourier_scale=fourier_scale,
            fourier_frequencies=fourier_frequencies,
            positive=positive,
            parameterization='cholesky',
            residual=True,
            name=name + "_SpdOperator",
        )

        self.input_scalings_dict_ = {}
        self.input_scalings_ = torch.ones(self.n_outputs_)
        self.output_scalings_ = torch.ones(self.n_outputs_)
        self.scalings_set_ = False

    def _scale_inputs(self, inputs):
        if not self.scalings_set_:
            return inputs
        scaled_inputs = dict(inputs)
        for input_arg in self.depends_on_names_:
            if input_arg in scaled_inputs:
                scaled_inputs[input_arg] = scaled_inputs[input_arg] / self.input_scalings_dict_[input_arg]
        if self.acts_on_name_ in scaled_inputs and self.acts_on_name_ not in self.depends_on_names_:
            scaled_inputs[self.acts_on_name_] = (
                scaled_inputs[self.acts_on_name_] / self.input_scalings_dict_[self.acts_on_name_]
            )
        return scaled_inputs

    def _net(self, inputs):
        x = inputs[self.acts_on_name_]
        spd_out = self.spd_operator_.forward(inputs)
        scalar = torch.sum(x * spd_out, dim=1, keepdim=True)
        return scalar

    def _grad(self, inputs):
        x = inputs[self.acts_on_name_].detach().requires_grad_(True)
        inputs_local = dict(inputs)
        inputs_local[self.acts_on_name_] = x
        y = torch.sum(self._net(inputs_local))
        grad, = torch.autograd.grad(y, x, create_graph=True)
        return grad

    def _hessian(self, inputs):
        x = inputs[self.acts_on_name_]
        hessians = []
        for i in range(x.shape[0]):
            xi = x[i]

            def scalar_fn(z):
                inputs_local = dict(inputs)
                inputs_local[self.acts_on_name_] = z.unsqueeze(0)
                return torch.sum(self._net(inputs_local))

            hess = torch.autograd.functional.hessian(scalar_fn, xi, create_graph=True)
            hessians.append(hess)
        return torch.stack(hessians, dim=0)

    def forward(self, inputs, return_jacobian=False):
        """
        Forward pass of operator

        Parameters
        ----------
        inputs : dict(str, np.array)
            Dictionary of input data in the form of arrays referenced by the
            variable name, i.e., inputs['x'] = np.ones(3)

        return_jacobian: bool, optional
            If True, return the (approximate) Jacobian in addition to the output.
        """
        scaled_inputs = self._scale_inputs(inputs)
        grad_scaled = self._grad(scaled_inputs)
        grad = grad_scaled * self.output_scalings_ 
        if return_jacobian:
            hess_scaled = self._hessian(scaled_inputs)
            row_scale = (self.output_scalings_).unsqueeze(0).unsqueeze(2)
            col_scale = (1.0 / self.input_scalings_).unsqueeze(0).unsqueeze(1)
            jac = row_scale * hess_scaled * col_scale
            return grad[:, :], jac
        else:
            return grad[:, :]

    def set_scalings(self, input_scalings_dict, output_scaling):
        with torch.no_grad():
            self.scalings_set_ = True
            self.input_scalings_dict_ = {}
            for input_arg in self.depends_on_names_:
                self.input_scalings_dict_[input_arg] = input_scalings_dict[input_arg]
            if self.acts_on_name_ not in self.input_scalings_dict_:
                self.input_scalings_dict_[self.acts_on_name_] = input_scalings_dict[self.acts_on_name_]
            self.input_scalings_[:] = self.input_scalings_dict_[self.acts_on_name_]
            self.output_scalings_[:] = output_scaling


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
        n_inputs_total = 0
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())
          n_inputs_total += depends_on[i].get_size()
        self.has_affine_inputs_ = (n_inputs_total > 0)
        self.n_inputs_ = (n_inputs_total if self.has_affine_inputs_ else 1)
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
  
      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
  
      """

      # Separate out states and parameters
      x = inputs[self.acts_on_name_]
      if self.has_affine_inputs_:
        mu = inputs_to_tensor(inputs,self.depends_on_names_)
      else:
        mu = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
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
      if self.has_affine_inputs_:
        input_scalings = None
        for input_arg in self.depends_on_names_:
          if input_scalings is None:
            input_scalings = input_scalings_dict[input_arg]
          else:
            input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      else:
        input_scalings = torch.ones_like(self.T[0, 0, :])
      
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
        n_inputs_total = 0
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())
          n_inputs_total += depends_on[i].get_size()
        self.has_affine_inputs_ = (n_inputs_total > 0)
        self.n_inputs_ = (n_inputs_total if self.has_affine_inputs_ else 1)

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
  
      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
      """
      assert return_jacobian == False, "Return jacobian currently not implemented for linear skew operator"
      # Separate out states and parameters
      x = inputs[self.acts_on_name_]
      if self.has_affine_inputs_:
        mu = inputs_to_tensor(inputs,self.depends_on_names_)
      else:
        mu = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
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
      if self.has_affine_inputs_:
        input_scalings = None
        for input_arg in self.depends_on_names_:
          if input_scalings is None:
            input_scalings = input_scalings_dict[input_arg]
          else:
            input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      else:
        input_scalings = torch.ones_like(self.input_scalings_)
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
        n_inputs_total = 0
        for i in range(len(depends_on)):
          self.depends_on_names_.append(depends_on[i].get_name())
          n_inputs_total += depends_on[i].get_size()
        self.has_affine_inputs_ = (n_inputs_total > 0)
        self.n_inputs_ = (n_inputs_total if self.has_affine_inputs_ else 1)

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


    def forward(self,inputs, return_jacobian=False):
      """
      Forward pass of operator 

      .. warning:: return_jacobian is currently not implemented 
  
      Parameters
      ----------
      inputs : dict(str, np.array) 
          Dictionary of input data in the form of arrays referenced by the variable name, i.e., inputs['x'] = np.ones(3)
  
      return_jacobian: bool, optional
          If True, return the (approximate) Jacobian in addition to the output.
      """
      assert return_jacobian == False, "Return jacobian currently not implemented for linear skew operator"

      # Separate out states and parameters
      x = inputs[self.acts_on_name_]
      if self.has_affine_inputs_:
        mu = inputs_to_tensor(inputs,self.depends_on_names_)
      else:
        mu = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
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
      if self.has_affine_inputs_:
        input_scalings = None
        for input_arg in self.depends_on_names_:
          if input_scalings is None:
            input_scalings = input_scalings_dict[input_arg]
          else:
            input_scalings = torch.cat( (input_scalings,input_scalings_dict[input_arg]),0)
      else:
        input_scalings = torch.ones_like(self.input_scalings_)
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
