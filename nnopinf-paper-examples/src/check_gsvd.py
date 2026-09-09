import numpy as np
import scipy

A = np.random.normal(size=(30,10))
M = np.random.normal(size=(30,30))
M = M @ M.transpose() 

Mc = np.linalg.cholesky(M).transpose()
assert np.allclose(Mc.transpose() @ Mc, M)
u,s,v = np.linalg.svd(Mc @ A,full_matrices=False)
uc = u
uc = np.linalg.solve(Mc, u)

print(uc.transpose() @ M @ uc)
