# The function mysolve(A, b) is invoked by ccore.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ccore.py -clscale float
# where the line argument 'clscale' allows the global coarsening of the mesh
# -clscale 1  fine mesh
# -clscale 10 coarse mesh


# Homework 1 : QR Factorization with Householder reflectors
# @author Gilles Peiffer, 24321600
# gilles.peiffer@student.uclouvain.be
# Oct. 23, 2018


SolverType = 'QR'

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def mysolve(A, b):
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)  # Needs import
    elif SolverType == 'QR':
        return True, QRsolve(A, b)
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

    A = np.array(A, dtype=np.float64)
    m, n = A.shape
    V = np.zeros((m, n))  # Stores the different reflection vectors
    
    def new_sign(x):
        if x >= 0:
            return 1
        return -1
    
    for k in range(n):
        e1 = np.zeros(m-k)
        e1[0] = 1
        x = A[k:m, k]
        V[k:m, k] = new_sign(x[0])*np.linalg.norm(x, 2)*e1 + x[:]
        V[k:m, k] /= np.linalg.norm(V[k:m, k], 2)  # Normed Householder reflector
        A[k:m, k:n] -= 2.* np.outer(V[k:m, k], np.dot(V[k:m, k], A[k:m, k:n]))
        
    return V, A


def QRsolve(A, b):
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
    
    b = np.array(b, dtype=np.float64)
    m, n = A.shape
    assert(m >= n), "Wrong matrix dimensions."
    assert(len(b) == m), "Matrix and vector dimensions not aligned."
    
    V, R = QR(A)
    
    x = np.zeros(m)
    
    # Compute the product Qb
    for k in range(n):
        b[k:m] -= 2.* np.dot(V[k:m, k], np.dot(V[k:m, k], b[k:m]))
        
    # Back substitution to find x
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            b[i] -= R[i, j]*x[j]
        x[i] = b[i]/R[i, i]
        
    return x


def plot_complexity(precision='report'):
    """
    This function plots various graphs that give more insight
    into how the QRsolve function works
    and how it compares to the built-in solver in the NumPy library.
    
    The default argument is for report-quality graphs,
    if the precision parameter is set to 'test',
    it executes much quicker but doesn't give nice graphs.
    """
    if precision == 'test':
        N = range(20, 400, 20)
    elif precision == 'report':
        N = range(20, 1000, 10)
    T = []
    V = []
    for n in N:
        A = np.random.rand(n, n)  # Random matrix
        b = np.random.random(n)  # Random vector
        start_time_QR = timer()
        QRsolve(A,b)
        t_QR = timer()-start_time_QR
        T.append(t_QR)
        start_time_NumPy = timer()
        np.linalg.solve(A,b)
        t_NumPy = timer()-start_time_NumPy
        V.append(t_NumPy)
        
    logN = np.log(N)
    logT = np.log(T)
    logV = np.log(V)
        
    # Linear regression
    fit_QR = np.polyfit(logN, logT, 1)
    fit_NumPy = np.polyfit(logN, logV, 1)
    fit_fn_QR = np.poly1d(fit_QR)
    fit_fn_NumPy = np.poly1d(fit_NumPy)
    
    # Cubic fit
    fit_QR3 = np.polyfit(N, T, 3)
    fit_fn_QR3 = np.poly1d(fit_QR3)
    
    # Logarithmic plot of QRsolve execution time
    plt.plot(logN, logT, 'bo', logN, fit_fn_QR(logN), '--b')
    plt.xlabel(r"$\log(n)$")
    plt.ylabel(r"$\log(t)$")
    plt.title(r"Execution time of QRsolve on a logarithmic scale"
              "\n"
              r"as a function of $n$, the size of $A$")
    plt.show()
    
    # Logarithmic plot of NumPy execution time
    plt.plot(logN, logV, 'ro', logN, fit_fn_NumPy(logN), '--r')
    plt.xlabel(r"$\log(n)$")
    plt.ylabel(r"$\log(t)$")
    plt.title(r"Execution time of np.linalg.solve on a logarithmic scale"
              "\n"
              r"as a function of $n$, the size of $A$")
    plt.show()
    
    # Regular plot of QRsolve execution time
    plt.plot(N, T, 'go', N, fit_fn_QR3(N), '--g')
    plt.title(r"Execution time of QRsolve"
              "\n"
              r" as a function of $n$, the size of $A$")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$t$")
    plt.show()


if __name__ == "__main__":
    plot_complexity('test')