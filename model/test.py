import torch
import numpy as np

A = np.array([0, 1, 2])
B = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]])
print(np.einsum('i,ij->i', A, B))
print(np.einsum('i,ij->j', A, B))
