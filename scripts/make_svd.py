import numpy as np
import sys

input_file='user_biases'
n=10
x=np.load(input_file + '.npy')
u, d, v = np.linalg.svd(x, full_matrices=False)
np.save(input_file + '_sing_vals',d)
u=u[:,:n]
v=v[:n, :].T
np.save(input_file + '_u', u)
np.save(input_file + '_v', v)
