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
        i, sol = low_rank_approx(A,b)
        return True, scipy.sparse.linalg.spsolve(A, b), i
    else:
        return False, 0

def low_rank_approx(A,b):
     U, s, Vh = np.linalg.svd(A)
     sol = 0
     for i in range(len(s)):
         sol += np.dot(b,(np.outer(U.T[i], Vh[i])/s[i]))
     pourcentage = 0
     i =1
     sol2=0
     while i < len(sol) and pourcentage < 0.9 :
         sol2+=np.dot(b,(np.outer(U.T[len(sol)-i], Vh[len(sol)-i])/s[len(sol)-i]))
         pourcentage = (np.linalg.norm(sol) - np.linalg.norm(sol-sol2))/np.linalg.norm(sol)
         first = np.dot(b,(np.outer(U.T[len(sol)-1], Vh[len(sol)-1])/s[len(sol)-1]))
         first1 =  (np.linalg.norm(sol) - np.linalg.norm(sol-first))/np.linalg.norm(sol)
         prc = np.linalg.norm(np.dot(b,(np.outer(U.T[len(sol)-1], Vh[len(sol)-1]/s[len(sol)-1]))))/np.linalg.norm(sol)
         
         i+=1
     #print(first1)
     #print(prc)
     #print(i)
     #print(i-1)
     return first1, sol