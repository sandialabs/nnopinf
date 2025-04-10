import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
torch.set_default_dtype(torch.float64)


class CompositeOperator(nn.Module):
    def __init__(self,state_operators):
        super(CompositeOperator, self).__init__()
        self.state_operators =  nn.ModuleList(state_operators)

    def forward(self,x,return_stiffness=False):
        if return_stiffness:
            result,stiffness = self.state_operators[0].forward(x,return_stiffness=True)
            for state_operator in self.state_operators[1::]:
                t1,t2 = state_operator.forward(x,return_stiffness=True)
                result += t1
                stiffness += t2
            return result,stiffness
        else:
            result = self.state_operators[0].forward(x)
            for state_operator in self.state_operators[1::]:
                result += state_operator.forward(x)
            return result 

    def set_scalings(self,input_scaling,output_scaling):
        for operator in self.state_operators:
            operator.set_scalings(input_scaling,output_scaling)
 
    def hierarchical_update(self,input_operator):
        assert len(input_operator.state_operators) == len(self.state_operators)
        for i in range(len(self.state_operators)):
            operator = self.state_operators[i]
            operator.hierarchical_update(input_operator.state_operators[i])


class NpdOperator(nn.Module):
    def __init__(self,n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs):
        super(NpdOperator, self).__init__()
        self.SpdOperator = SpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
        self.forward_list = self.SpdOperator.forward_list

    def forward(self,x,return_stiffness=False):
        if return_stiffness:
            result,stiffness = self.SpdOperator.forward(x,return_stiffness=True)
            return -result,-stiffness
        else:
            result = self.SpdOperator.forward(x)
            return -result

    def set_scalings(self,input_scaling,output_scaling):
        self.SpdOperator.set_scalings(input_scaling,output_scaling) 

    def hierarchical_update(self,input_operator):
        self.SpdOperator.hierarchical_update(input_operator)

class SpdOperator(nn.Module):
    def __init__(self,n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs):
        super(SpdOperator, self).__init__()
        forward_list = []
        idx = np.tril_indices(n_outputs)
        self.idx = idx

        self.n_outputs = n_outputs
        networkOutputSize = idx[0].size
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        dim = np.zeros(n_hidden_layers+2,dtype='int')
        dim[0] = n_inputs
        for i in range(1,n_hidden_layers+1):
         dim[i] = n_neurons_per_layer
        dim[-1] = networkOutputSize
        self.dim = dim
        input_dim = dim[0:-1]
        output_dim = dim[1::]

        for i in range(0,n_hidden_layers+1):
          forward_list.append(nn.Linear(input_dim[i], output_dim[i]))
        self.forward_list = nn.ModuleList(forward_list)
        self.activation = F.relu
        self.input_scaling_ = torch.ones(n_inputs)
        self.output_scaling_ = torch.ones(n_outputs)
        self.input_scaling_mat_ = torch.eye(n_outputs)     
        self.output_scaling_mat_ = torch.eye(n_outputs)     

    def forward(self,x,return_stiffness=False):
      y = x/self.input_scaling_[None] 
      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))

      y = self.forward_list[-1](y)

      K = torch.zeros(y.shape[0],self.n_outputs,self.n_outputs)

      K[:,self.idx[0],self.idx[1]] = y[:,0:self.idx[0].size]

      KT = torch.transpose(K,2,1)
      K = torch.einsum('ijk,ikl->ijl',K,KT)
      K = torch.einsum('ij,njk->nik',self.output_scaling_mat_,K) 
      K = torch.einsum('nij,jk->nik',K,self.input_scaling_mat_) 

      result = torch.einsum('ijk,ik->ij',K,x[:,0:self.n_outputs] )
      if return_stiffness:
          return result, K
      else:
          return result

    def set_scalings(self,input_scaling,output_scaling):
      n_outputs = self.output_scaling_.size()[0]
      self.input_scaling_[:] = input_scaling
      self.output_scaling_[:] = output_scaling
      self.output_scaling_mat_[:] = torch.eye(output_scaling.size()[0])*output_scaling
      self.input_scaling_mat_[:] = torch.eye(n_outputs)/input_scaling[0:n_outputs]

    def hierarchical_update(self,input_operator):
      assert len(input_operator.forward_list) == len(self.forward_list)

      for i in range(1,self.num_layers-1):
        assert self.forward_list[i].weight.shape == input_operator.forward_list[i].weight.shape
        assert self.forward_list[i].bias.shape == input_operator.forward_list[i].bias.shape

      with torch.no_grad():
        input_weights = input_operator.forward_list[0].weight
        input_bias = input_operator.forward_list[0].bias
        assert input_bias.size() == self.forward_list[0].bias.size(), " Incompatable shapes"
        self.forward_list[0].weight[:,0:input_weights.shape[1]] = input_weights[:,:]
        #self.forward_list[0].weight[:,input_weights.shape[1]::] = 0.#input_weights[:,:]
        self.forward_list[0].bias[:] = input_bias[:]
        for i in range(1,self.num_layers-1):
          self.forward_list[i].weight[:] = input_operator.forward_list[i].weight[:]
          self.forward_list[i].bias[:] = input_operator.forward_list[i].bias[:]
          #self.forward_list[i].weight.requires_grad = False
          #self.forward_list[i].bias.requires_grad = False

        final_weights = input_operator.forward_list[-1].weight[:]
        final_bias = input_operator.forward_list[-1].bias[:]
        self.forward_list[-1].weight[0:final_weights.shape[0],:] = final_weights[:]
        self.forward_list[-1].bias[0:final_weights.shape[0]] = final_bias[:]
        #self.forward_list[-1].weight[final_weights.shape[0]::,:] = 0. 
        #self.forward_list[-1].bias[final_weights.shape[0]::] = 0. 


class MatrixOperator(nn.Module):
    def __init__(self,n_hidden_layers,n_neurons_per_layer,n_inputs,output_shape):
        super(MatrixOperator, self).__init__()
        forward_list = []

        self.output_shape = output_shape
        networkOutputSize = output_shape[0]*output_shape[1] 
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        dim = np.zeros(n_hidden_layers+2,dtype='int')
        dim[0] = n_inputs
        for i in range(1,n_hidden_layers+1):
         dim[i] = n_neurons_per_layer
        dim[-1] = networkOutputSize
        self.dim = dim
        input_dim = dim[0:-1]
        output_dim = dim[1::]

        for i in range(0,n_hidden_layers+1):
          forward_list.append(nn.Linear(input_dim[i], output_dim[i]))

        self.forward_list = nn.ModuleList(forward_list)
        self.activation = F.relu
        self.input_scaling_ = torch.ones(n_inputs)
        self.output_scaling_ = torch.ones(output_shape[0])
        self.output_scaling_mat_ = torch.eye(output_shape[0])*self.output_scaling_ 
        self.input_scaling_mat_ = torch.eye(output_shape[1])

    def forward(self,x,return_stiffness=False):
      y = x/self.input_scaling_[None] 
      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))
      y = self.forward_list[-1](y)
      A = torch.zeros(y.shape[0],self.output_shape[0],self.output_shape[1])
      A[:] = torch.reshape(y,A.shape)
      A = torch.einsum('ij,njk->nik',self.output_scaling_mat_,A) 
      A = torch.einsum('nij,jk->nik',A,self.input_scaling_mat_) 
      result = torch.einsum('ijk,ik->ij',A,x[:,0:self.output_shape[1]])
      if return_stiffness:
          return result[:,:],A
      else:
          return result[:,:]

    def set_scalings(self,input_scaling,output_scaling):
      n_inputs = self.output_shape[1]
      self.input_scaling_[:] = input_scaling
      self.output_scaling_[:] = output_scaling
      self.output_scaling_mat_[:] = torch.eye(output_scaling.size()[0])*output_scaling
      self.input_scaling_mat_[:] = torch.eye(n_inputs)/input_scaling[0:n_inputs]


    def hierarchical_update(self,input_operator):
      assert len(input_operator.forward_list) == len(self.forward_list)

      for i in range(1,self.num_layers-1):
        assert self.forward_list[i].weight.shape == input_operator.forward_list[i].weight.shape
        assert self.forward_list[i].bias.shape == input_operator.forward_list[i].bias.shape

      with torch.no_grad():
        input_weights = input_operator.forward_list[0].weight
        input_bias = input_operator.forward_list[0].bias
        assert input_bias.size() == self.forward_list[0].bias.size(), " Incompatable shapes"
        self.forward_list[0].weight[:,0:input_weights.shape[1]] = input_weights[:,:]
        self.forward_list[0].bias[:] = input_bias[:]
        for i in range(1,self.num_layers-1):
          self.forward_list[i].weight[:] = input_operator.forward_list[i].weight[:]
          self.forward_list[i].bias[:] = input_operator.forward_list[i].bias[:]

        final_weights = input_operator.forward_list[-1].weight[:]
        final_bias = input_operator.forward_list[-1].bias[:]
        self.forward_list[-1].weight[0:final_weights.shape[0],:] = final_weights[:]
        self.forward_list[-1].bias[0:final_weights.shape[0]] = final_bias[:]


class SkewOperator(nn.Module):
    def __init__(self,n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs):
        super(SkewOperator, self).__init__()
        forward_list = []
        idx = np.tril_indices(n_outputs)
        self.idx = idx

        self.n_outputs = n_outputs
        networkOutputSize = idx[0].size
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        dim = np.zeros(n_hidden_layers+2,dtype='int')
        dim[0] = n_inputs
        for i in range(1,n_hidden_layers+1):
         dim[i] = n_neurons_per_layer
        dim[-1] = networkOutputSize
        self.dim = dim
        input_dim = dim[0:-1]
        output_dim = dim[1::]

        for i in range(0,n_hidden_layers+1):
          forward_list.append(nn.Linear(input_dim[i], output_dim[i]))

        self.forward_list = nn.ModuleList(forward_list)
        self.activation = F.relu
        self.input_scaling_ = torch.ones(n_inputs)
        self.output_scaling_ = torch.ones(n_outputs)
        self.output_scaling_mat_ = torch.eye(n_outputs)*self.output_scaling_ 
        self.input_scaling_mat_ = torch.eye(n_outputs)     


    def forward(self,x,return_stiffness=False):
      y = x/self.input_scaling_[None] 
      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))

      y = self.forward_list[-1](y)

      S = torch.zeros(y.shape[0],self.n_outputs,self.n_outputs)

      S[:,self.idx[0],self.idx[1]] = y[:,0:self.idx[0].size]

      ST = torch.transpose(S,2,1)
      R = S - ST 
      # Apply scaling to R so matrix has correct scaling
      R = torch.einsum('ij,njk->nik',self.output_scaling_mat_,R) 
      R = torch.einsum('nij,jk->nik',R,self.input_scaling_mat_) 
      result = torch.einsum('ijk,ik->ij',R,x[:,0:self.n_outputs])
      if return_stiffness:
          return result[:,:],R
      else:
          return result[:,:]

    def set_scalings(self,input_scaling,output_scaling):
      n_outputs = self.output_scaling_.size()[0]
      self.input_scaling_[:] = input_scaling
      self.output_scaling_[:] = output_scaling
      self.output_scaling_mat_[:] = torch.eye(output_scaling.size()[0])*output_scaling
      self.input_scaling_mat_[:] = torch.eye(n_outputs)/input_scaling[0:n_outputs]

    def hierarchical_update(self,input_operator):
      assert len(input_operator.forward_list) == len(self.forward_list)

      for i in range(1,self.num_layers-1):
        assert self.forward_list[i].weight.shape == input_operator.forward_list[i].weight.shape
        assert self.forward_list[i].bias.shape == input_operator.forward_list[i].bias.shape

      with torch.no_grad():
        input_weights = input_operator.forward_list[0].weight
        input_bias = input_operator.forward_list[0].bias
        assert input_bias.size() == self.forward_list[0].bias.size(), " Incompatable shapes"
        self.forward_list[0].weight[:,0:input_weights.shape[1]] = input_weights[:,:]
        self.forward_list[0].bias[:] = input_bias[:]
        #self.forward_list[0].weight[:,input_weights.shape[1]::] = 0.#input_weights[:,:]
        self.forward_list[0].bias[:] = input_bias[:]

        for i in range(1,self.num_layers-1):
          self.forward_list[i].weight[:] = input_operator.forward_list[i].weight[:]
          self.forward_list[i].bias[:] = input_operator.forward_list[i].bias[:]
          #self.forward_list[i].weight.requires_grad = False
          #self.forward_list[i].bias.requires_grad = False

        final_weights = input_operator.forward_list[-1].weight[:]
        final_bias = input_operator.forward_list[-1].bias[:]
        self.forward_list[-1].weight[0:final_weights.shape[0],:] = final_weights[:]
        self.forward_list[-1].bias[0:final_weights.shape[0]] = final_bias[:]
        #self.forward_list[-1].weight[final_weights.shape[0]::,:] = 0. 
        #self.forward_list[-1].bias[final_weights.shape[0]::] = 0. 


class StandardOperator(nn.Module):
    def __init__(self,n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs):
        super(StandardOperator, self).__init__()
        forward_list = []
        self.n_outputs = n_outputs
        self.num_hidden_layers = n_hidden_layers
        self.num_layers = self.num_hidden_layers + 1

        dim = np.zeros(n_hidden_layers+2,dtype='int')
        dim[0] = n_inputs
        for i in range(1,n_hidden_layers+1):
         dim[i] = n_neurons_per_layer
        dim[-1] = self.n_outputs 
        self.dim = dim
        input_dim = dim[0:-1]
        output_dim = dim[1::]

        for i in range(0,n_hidden_layers+1):
          forward_list.append(nn.Linear(input_dim[i], output_dim[i]))

        self.forward_list = nn.ModuleList(forward_list)

        self.activation = F.relu
        self.input_scaling_ = torch.ones(n_inputs)
        self.output_scaling_ = torch.ones(n_outputs)

    def set_scalings(self,input_scaling,output_scaling):
      self.input_scaling_[:] = input_scaling
      self.output_scaling_[:] = output_scaling

    def forward(self,x):
      y = x/self.input_scaling_[None] 
      for i in range(0,self.num_layers-1):
        y = self.activation(self.forward_list[i](y))
      result = self.forward_list[-1](y)
      result = result*self.output_scaling_[None]
      return result[:,:]


    def hierarchical_update(self,input_operator):
      assert len(input_operator.forward_list) == len(self.forward_list)

      for i in range(1,self.num_layers-1):
        assert self.forward_list[i].weight.shape == input_operator.forward_list[i].weight.shape
        assert self.forward_list[i].bias.shape == input_operator.forward_list[i].bias.shape

      with torch.no_grad():
        input_weights = input_operator.forward_list[0].weight
        input_bias = input_operator.forward_list[0].bias
        assert input_bias.size() == self.forward_list[0].bias.size(), " Incompatable shapes"
        self.forward_list[0].weight[:,0:input_weights.shape[1]] = input_weights[:,:]
        self.forward_list[0].bias[:] = input_bias[:]
        for i in range(1,self.num_layers-1):
          self.forward_list[i].weight[:] = input_operator.forward_list[i].weight[:]
          self.forward_list[i].bias[:] = input_operator.forward_list[i].bias[:]

        final_weights = input_operator.forward_list[-1].weight[:]
        final_bias = input_operator.forward_list[-1].bias[:]
        self.forward_list[-1].weight[0:final_weights.shape[0],:] = final_weights[:]
        self.forward_list[-1].bias[0:final_weights.shape[0]] = final_bias[:]


if __name__ == "__main__":
    n_hidden_layers = 2
    n_neurons_per_layer = 5
    n_inputs = 5
    n_outputs = 5
    inputs = torch.tensor(np.random.normal(size=(10,n_inputs)))
    parameters = torch.tensor(np.random.normal(size=(10,0)))

    
    StandardMlp = StandardOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SpdMlp = SpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    NpdMlp = NpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SkewMlp = SkewOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)

    ops = [StandardMlp,NpdMlp,SkewMlp]
    CompositeMlp = CompositeOperator(ops) 
    inputs = torch.cat((inputs,parameters),1)
    r1 = StandardMlp.forward(inputs)
    r2 = NpdMlp.forward(inputs)
    r3 = SkewMlp.forward(inputs)
    r4 = CompositeMlp(inputs)
    assert np.allclose( r4.detach().numpy(), (r1 + r2 + r3).detach().numpy())
