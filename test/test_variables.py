import nnopinf
import pytest
import numpy as np

def test_variables():
  name = 'myvar'
  my_variable = nnopinf.Variable(size=5,name=name,normalization_strategy='MaxAbs')
  assert(my_variable.get_name() == name)
  assert(my_variable.get_size() == 5)
  good_data = np.zeros((8,5))
  my_variable.set_data(good_data)
  bad_data = np.zeros((8,4))
  with pytest.raises(AssertionError):
    my_variable.set_data(bad_data)

  with pytest.raises(AssertionError):
    my_variable = nnopinf.Variable(size=5,name=name,normalization_strategy='MaxAbsBad')


