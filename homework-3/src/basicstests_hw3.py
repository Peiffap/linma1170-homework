# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:04:10 2018

@author: sebco
"""

import scipy.linalg
import numpy  as np
import lu

eps = 1.e-6

"""TEST LUfactorize"""
def testLUfactorize():
    print('Test LUfactorize')
    print('---------------------------------------')

    #Matrix
    n=100
    A = creatDF(10)
    
    #Scipy LU
    [P,L,U] = scipy.linalg.lu(np.copy(A))
    
    #your LU
    LU = np.copy(A)
    lu.LUfactorize(LU)
    
    Ltest = np.eye(n) + np.tril(LU,-1)
    Utest = np.triu(LU)
    
    Atest = Ltest @ Utest
    if(np.all(abs(Atest - A)<= eps)):
        print('LU decomposition is exact')
    else :
        print('Mistake found in LU decomposition')
    if(np.all(abs(Ltest - L)<= eps)):
        print('L is exact')
    else :
        #print(abs(Ltest - L))
        print('L not exact')
        
    if(np.all(abs(Utest - U)<= eps)):
        print('U is exact')
    else :
        #print(abs(Utest - U))
        print('U not exact')
    print('---------------------------------------\n')
    
"""Test LUsolve"""
def testLUsolve():
    print('Test LUsolve')
    print('---------------------------------------')
    n = 100
    A = creatDF(10)
    b = 5*np.random.rand(n)-10
    x = scipy.linalg.solve(np.copy(A),b)
    xtest = lu.LUsolve(np.copy(A),b)
    if(np.all( abs(xtest - x) <= eps)):
        print('Solver works correctly : solution x is exact')
    else :
        print('Solver does not work correctly : solution x is not exact')
    print('---------------------------------------\n')

"""TEST CSR Format"""
def testCSRformat():
    print('Test CSRformat')
    print('---------------------------------------')
    #1st Example
    A1 = np.zeros((4,4))
    [sA1,iA1,jA1] = lu.CSRformat(A1)
    iA1_sol = np.zeros(5,dtype = int)
    jA1_sol = np.zeros(0,dtype = int)
    sA1_sol = np.zeros(0,dtype = complex)
    if (np.all(iA1 == iA1_sol) and np.all(jA1 == jA1_sol) and np.all(sA1 == sA1_sol)):
        print('CSR Format is exact for example 1')
    else :
        print('CSR Format is not correct for example 1')
        print(iA1,jA1,sA1)
        print(iA1_sol,jA1_sol,sA1_sol)
    
    #2nd Example
    A2 = np.array([[1,2,0],[4,5,6],[0,0,1]])
    [sA2,iA2,jA2] = lu.CSRformat(A2)
    iA2_sol = np.array([0,2,5,6],dtype = int)
    jA2_sol = np.array([0,1,0,1,2,2],dtype = int)
    sA2_sol = np.array([1,2,4,5,6,1],dtype = complex)
    
    if (np.all(iA2 == iA2_sol) and np.all(jA2 == jA2_sol) and np.all(sA2 == sA2_sol)):
        print('CSR Format is exact for example 2')
    else :
        print('CSR Format is not correct for example 2')
        print(iA2,jA2,sA2)
        print(iA2_sol,jA2_sol,sA2_sol)
    print('---------------------------------------\n')

    
"""TEST LUCSR"""
def testLUcsr():
    print('Test LUcsr')
    print('---------------------------------------')
    
    #EX1 : small matrice - not sparse - not symmetric but positive definite
    A = np.array([[3.,-1.,1.],[-2.,2.,-1.],[0.,-1.,4.]])
    #print(A)
    #print(np.linalg.eigvals(A))

    #Your solution
    [sA,iA,jA] = lu.CSRformat(A)
    [sLU,iLU,jLU] = lu.LUcsr(sA,iA,jA)
    
    #True solution
    [P,L,U] = scipy.linalg.lu(A)
    LU_sol = np.tril(L,-1) + U
    [sLU_sol,iLU_sol,jLU_sol] = lu.CSRformat(LU_sol)
    #print(LU_sol)
    if (np.all(iLU == iLU_sol) and np.all(jLU == jLU_sol) and np.all(abs(sLU - sLU_sol)<=eps)):
        print('LUcsr is exact for example 1 : small matrice (not sparse - not symmetric) ')
    else:
        print(sLU_sol)
        print(sLU)
        print('LUcsr is not correct for example 1 : small matrice (not sparse  - not symmetric)')
        
    # EX2 : bigger matrix but sparse, symetric and positive definite
    A = creatDF(10) #create matrix 100*100

    #Your solution
    [sA,iA,jA] = lu.CSRformat(A)
    [sLU,iLU,jLU] = lu.LUcsr(sA,iA,jA)
    
    #True solution
    [P,L,U] = scipy.linalg.lu(A)
    LU_sol = np.tril(L,-1) + U
    [sLU_sol,iLU_sol,jLU_sol] = lu.CSRformat(LU_sol)
    
    if (np.all(iLU == iLU_sol) and np.all(jLU == jLU_sol) and np.all(abs(sLU - sLU_sol)<=eps)):
        print('LUcsr is exact for example 2 : bigger matrix but sparse and symmetric')
    else :
        print('LUcsr is not correct for example 2 : bigger matrix but sparse and symmetric')
    print('---------------------------------------\n')

"""
creatSymDF crée une matrice A de différences finies de taille n²xn² pour une problème de Laplace
@param : n taille de la discrétisation du problème de Laplace
@return : la partie symétrique de A, définie positive dimension n² x n²
"""
def creatDF(n):
    n2 = n * n
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


"""
density calcule la densité d'une matrice
@param : A est une matrice passée sous forme de numpy array
@return : densité de A (nnz/(n*n))
"""

def density(A):
    n = A.shape[0]
    return sum(sum(A!=0))/(n*n)


#TESTS
testLUfactorize()
testLUsolve()
testCSRformat()
testLUcsr()