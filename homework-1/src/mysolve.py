# The function mysolve(A, b) is invoked by ccore.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ccore.py -clscale float
# where the line argument 'clscale' allows the global coarsening of the mesh
# -clscale 1  fine mesh
# -clscale 10 coarse mesh


SolverType = 'scipy'

import scipy.sparse
import scipy.sparse.linalg
import numpy

def mysolve(A, b):
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)
    elif SolverType == 'QR':
        # TODO write QR solver code
    #elif SolverType == 'LU':
        # write here your code for the LU solver
    #elif SolverType == 'GMRES':
        # write here your code for the LU solver
    else:
        return False, 0
    

def QR(A):
    """
    Calculates part of the Householder QR factorization of a matrix A.
    
    Input:
        A: an m x n matrix.
        
    Output:
        V: an m x n matrix containing the reflection vectors
           used to create the Q matrix in A's QR factorization (A = QR).
           
        R: an upper-triangular matrix, it is the R matrix
           in A's QR factorization (A = QR).
           
        Both V and R are NumPy arrays.
    """
    V = []  # Stores the different reflection vectors
    
    e1 = np.zeros(n)  # Unit vector
    e1[0] = 1
    
    for k in range(n):
            x = np.array(A[k:m][k])
            vk = np.sign(x[0])*np.linalg.norm(x)*e1 + x
            vk = vk/np.linalg.norm(vk)
            V[k] = np.array(vk)
            A[k:m][k:n] -= 2*vk*(np.transpose(vk)*A[k:m][k:n])
    
    return V, R
            
        
