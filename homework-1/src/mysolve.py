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
import numpy as np

def mysolve(A, b):
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)
    elif SolverType == 'QR':
        return True, QRSolve(A, b)
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
    m = len(A)
    n = len(A[0])
    A = np.array(A, dtype=float)
    V = []  # Stores the different reflection vectors
    
    def find_unit_vector(l):
        e1 = np.reshape(np.zeros(l), (l, 1))  # Unit vector
        e1[0] = 1
        return e1
    
    for k in range(n):
        V.append([])
        x = np.reshape(A[k:m, k], (m-k, 1))
        vk = np.sign(x[0])*np.linalg.norm(x)*find_unit_vector(m-k) + x
        vk = vk/np.linalg.norm(vk)
        V[k] = vk
        A[k:m, k:n] = A[k:m, k:n] - 2.*np.dot(vk, np.dot(np.transpose(vk), A[k:m, k:n]))

    return V, A


def QRSolve(A, b):
    m = len(A)
    n = len(A[0])
    b = np.array(b, dtype=float)
    
    V, R = QR(A)
    x = np.reshape(np.zeros(n), (n, 1))
    
    def vecmatsum(xvec, Rmat, j):
        sum = 0
        for k in range(j+1, n):
            sum += xvec[k] * Rmat[j][k]
            
        return sum
    
    print(b)
    
    for k in range(n):
        b[k:m] = b[k:m] - 2.*np.dot(V[k], np.dot(np.transpose(V[k]), b[k:m]))
    
    print(b)
    
    for j in range(n):
        x[j] = (b[j] - vecmatsum(x, R, j))/R[j][j]
        
    return x


A = [[1., 1.], [0., 1.]]
b = [[2.], [1.]]

print(QRSolve(A, b))
print(scipy.sparse.linalg.spsolve(A, b))
            
        
