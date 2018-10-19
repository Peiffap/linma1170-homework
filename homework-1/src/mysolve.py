# The function mysolve(A, b) is invoked by ccore.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ccore.py -clscale float
# where the line argument 'clscale' allows the global coarsening of the mesh
# -clscale 1  fine mesh
# -clscale 10 coarse mesh


SolverType = 'QR'

import scipy.sparse
import scipy.sparse.linalg
import numpy as np

def mysolve(A, b):
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)
    elif SolverType == 'QR':
        return True, QRSolve(A, b)
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
    V = np.zeros((m, n))  # Stores the different reflection vectors
    
    def find_unit_vector(l):
        e1 = np.zeros(l)  # Unit vector
        e1[0] = 1
        return e1
    
    def new_sign(x):
        if x == 0:
            return 1
        return np.sign(x)
    
    for k in range(n-1):
        x = A[k:m, k]
        vk = new_sign(x[0])*np.linalg.norm(x)*find_unit_vector(len(x)) + x
        vk = vk/np.linalg.norm(vk)
        vk.shape = (vk.shape[0], 1)
        A[k:m, k:n] = A[k:m, k:n] - 2.* vk * (vk.T @ A[k:m, k:n])
        vk.shape = vk.shape[0]
        V[k:m, k] = vk
        
    return V, A


def QRSolve(A, b):
    """
    This function calculates the solution of a linear system of equations
    given under the form Ax = b. It uses QR factorization
    with Householder reflectors to factorize the matrix A,
    then uses the reflectors stored in V,
    as well as the upper-triangular matrix R,
    to back-substitute and find x.
    
    Input:
        A: the first matrix in Ax = b
        B: the second matrix in Ax = b
        
    Output:
        x: the solution to Ax = b, namely x = A^{-1}b = R^{-1}Q*b.
    """
    m = len(A)
    n = len(A[0])
    b = np.array(b, dtype=float)
    
    V, R = QR(A)
    
    V = np.array(V, dtype=float)
    
    x = np.zeros((n, 1))
    
    def vecmatsum(xvec, Rmat, j):
        sum = 0
        for k in range(j+1, len(xvec)):
            sum += xvec[k] * Rmat[j][k]
            
        return sum
    
    for k in range(n):
        vk = np.reshape(V[k:m, k], (m-k, 1))
        b[k:m] = b[k:m] - 2.* vk @ (np.transpose(vk) @ b[k:m])
        
    
    for i in range(n):
        j = n-i-1
        x[j] = (b[j] - vecmatsum(x, R, j))/R[j][j]
        
    return np.reshape(x, (1, len(A[0]))).flatten()


A = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1)

R1 = QRSolve(A, b)
R2 = scipy.sparse.linalg.spsolve(A, b)

print(np.allclose(R1, R2, 1e-5, 1e-5))

#print("QRSolve:", , "\n")
#print("SciPy:", , "\n")
            
        
