# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:05:40 2018

@author: HENRY Michel et COLLA Sebastien
"""

import numpy as np
import scipy.sparse.linalg as sc
import csrCG as CG

def creatDF(n):
    n2 = n*n
    A = np.zeros((n2,n2))
    for i in range(1,n+1):
        for j in range(1,n+1):
            index = i + (j-1)*n - 1

            if i==1 or i==n or j==1 or j==n :
                A[index,index] = 1.
            else :
                A[index,index] = 4.
                A[index,index+1] = -1.
                A[index,index-1] = -1.
                A[index,index+n] = -1.
                A[index,index-n] = -1.
    return A @ A.T
def CSRformat(A):
    (rows,jA) = A.nonzero()
    sA = A[(rows,jA)]
    iA = np.zeros(A.shape[0]+1,dtype=np.int)
    iA[1:] = np.cumsum(np.sum(A!=0, 1))
    return (sA,iA,jA)

def testCG():
    n = 3
    print("Test gradient conjugé sans préconditionneur avec n = ",n*n)
    print("------------------------ ")
    A = creatDF(n)
    b = np.ones(n*n)
    rtol = 1e-6
    
    # Solution avec Scipy sans préconditionneur
    x = np.zeros(n*n)
    x = sc.cg(A,b,tol=rtol)
    print("Solution via CG de Scipy \n x : ",x[0])

    prec = False
    sA,iA,jA = CSRformat(A)
    (x_test, res) = CG.csrCG(sA,iA,jA,b,rtol,prec)
    print("Solution via votre CG \n x : ",x_test)
    
    if(np.linalg.norm(x_test - x[0])/np.linalg.norm(b) <= rtol):
        print("Test réussi !!!")
    else :
        print("Tu diverges, verges...")
    print("------------------------\n")
    
    
def testPCG(): 
    n = 3
    A = creatDF(n)
    b = np.ones(n*n)
    rtol = 1e-6
    print("Test gradient conjugé avec préconditionneur avec n = ",n*n)
    print("------------------------ ")
    x = sc.cg(A,b,tol=rtol)
    print("Solution via CG de Scipy \n x : ",x[0])

    prec = True
    sA,iA,jA = CSRformat(A)
    (x_test, res) = CG.csrCG(sA,iA,jA,b,rtol,prec)
    print("Solution via votre CG \n x : ",x_test)
    
    
    if(np.linalg.norm(x_test - x[0])/np.linalg.norm(b) <= rtol):
        print("Test réussi !!!")
    else :
        print("Tu diverges, verges...")
    print("------------------------\n")
    
testCG()
testPCG()

    
    
    
    
    