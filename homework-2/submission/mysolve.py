#The function mysolve(A, b) is invoked by ccore.py
#to solve the linear system
#Implement your solver in this file and then run:
#python ccore.py -clscale float
#where the line argument 'clscale' allows the global coarsening of the mesh
#-clscale 1  fine mesh
#-clscale 10 coarse mesh

# Homework 2 : SVD and condition number
# @author Gilles Peiffer, 24321600
# gilles.peiffer@student.uclouvain.be
# Nov. 12, 2018


SolverType = 'scipy'

import scipy.sparse
import scipy.sparse.linalg
import numpy as np

def mysolve(A, b):
    if SolverType == 'scipy':
        nterms, first, sol = low_rank_approx(A,b)
        return True, scipy.sparse.linalg.spsolve(A, b), nterms, first
    else:
        return False, 0

def low_rank_approx(A,b):
     sol = 0    
     solApp = 0
     percent = 0
     Etot = 0
     U, s, Vh = np.linalg.svd(A)
     for i in range(len(s)):
         sol += np.dot(b,(np.outer(U.T[i], Vh[i])/s[i]))
     # necessary number of terms
     i = 1
     while i < len(sol) and percent < 0.9:
         solApp += np.dot(b, (np.outer(U.T[len(sol) - i], Vh[len(sol) - i])/s[len(sol) - i]))
         percent = (np.linalg.norm(sol) - np.linalg.norm(sol - solApp))/np.linalg.norm(sol)
         i += 1
     # energy calculation
     Etot = sum(np.divide(1, s))
     first = (1/s[len(s) - 1])/Etot
     return i - 1, first, sol