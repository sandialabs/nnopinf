import numpy as np

class Variable:
    def __init__(self,size,name,normalization_strategy='MaxAbs'):
        acceptable_normalization_strategies = ['Abs','MaxAbs']
        assert normalization_strategy in acceptable_normalization_strategies, "Error, acceptable normalization strategies are " + str(acceptable_normalization_strategies) 
        self.size_ = size
        self.name_ = name
        self.normalization_strategy_ = normalization_strategy
        self.data_ = np.zeros((0,self.size_))

    def get_name(self):
        return self.name_

    def get_size(self):
        return self.size_

    def set_data(self,data):
        assert data.shape[1] == self.size_, f"Error, variable {self.name_} is of size {self.size_}, input data is of size {data.shape}"
        
        self.data_ = data

