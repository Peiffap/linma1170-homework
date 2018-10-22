# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:10:29 2018

@author: Marie Hartman
@noma: 36801600

"""

import numpy as np
import matplotlib.pyplot as plt
import time

def QRfactorize(A):
    """
    Implémentation de l'algorithme d'Householder réalisant la décompostion QR. QRfactorize renvoit
    deux matrices w et R.
    
    INPUT:
    - A, une matrice de taille mxn tel que m>=n
    OUTPUTS:
    - w, une matrice de taille mxn qui contient tous les vecteurs v normés permettant de calculer Q
    - R, une matrice triangulaire supérieure de taille nxn.
    """
    def s(x):
        """
        La fonction s renvoit une indication sur le signe du nombre x.
        
        INPUT:
            - x, scalaire représentant la valeur d'entrée
        OUTPUT:
            - le signe de x. La fonction renvoit 1 si x>=0, sinon elle renvoit -1
        """
        if x>= 0:
            return 1
        else:
            return -1
    A = np.array(A, dtype=np.float64)
    m,n = A.shape
    R = np.copy(A)
    w = np.zeros((m,n))
    for k in range(n):
        e1 = np.zeros(m-k)
        e1[0] = 1
        x = R[k:m,k] 
        w[k:m,k] = s(x[0]) * np.linalg.norm(x,2)*e1 + x[:]
        w[k:m,k] = w[k:m,k]/np.linalg.norm(w[k:m,k],2)
        R[k:m,k:n] = R[k:m,k:n] - 2*np.outer(w[k:m,k],np.dot(w[k:m,k],R[k:m,k:n]))
        
    return w,R
def QRsolve(A,b):
    """
    Implémentation de l'algorithme "Backward Subsitution" permettant de résoudre le système 
    linéaire Ax = b.La fonction fait également appel à QRfactorization pour la décomposition QR de A
    La fonction QRsolve renvoit la valeur du vecteur x.
        
    INPUTS:
       - A, une matrice de taille nxn
       - b, un vecteur de taille nx1
    OUTPUT:
        - x, le vecteur colonne de taille nx1
    """
    m,n = A.shape
    x = np.zeros(m)
    W,R = QRfactorize(A)
    b = np.array(b, dtype=np.float64)
    
    for k in range(n):
        b[k:m] = b[k:m] - 2*np.dot(W[k:m,k],np.dot(W[k:m,k],b[k:m]))
    for i in range(n-1,-1,-1):
        for j in range(i+1,n):
            b[i] =b[i] - R[i,j]*x[j]
        x[i] = b[i]/R[i,i]
        
    return x

N = []
T = []
C = []
V = []
for n in range(200,1000,20):
    A = np.random.rand(n,n)
    b = np.random.random(n)
    start_time_QR = time.perf_counter()
    x = QRsolve(A,b)
    t_QR = time.perf_counter()-start_time_QR
    N.append(n)
    T.append(t_QR)
    start_time_SciPy = time.perf_counter()
    y = np.linalg.solve(A,b)
    t_SciPy = time.perf_counter()-start_time_SciPy
    C.append(n)
    V.append(t_SciPy)
    
#Regression linéaire
fit_QR = np.polyfit(np.log(N),np.log(T),1)
fit_SciPy = np.polyfit(np.log(C),np.log(V),1)
fit_fn_QR =np.poly1d(fit_QR)
fit_fn_SciPy = np.poly1d(fit_SciPy)
#Regression cubique
fit_QR3 = np.polyfit(N,T,3)
fit_fn_QR3 =np.poly1d(fit_QR3)

#plot logarithme QRsolve
plt.plot(np.log(N),np.log(T), 'bo',np.log(N),fit_fn_QR(np.log(N)),'--k')
plt.xlabel('log(n)')
plt.ylabel('log(t)')
plt.title("Temps d'exécution en fonction de la taille n \n en échelle logarithmique de QRsolve" )
plt.show()
#plot logarithme SciPy
plt.plot(np.log(C),np.log(V),'ro',np.log(C),fit_fn_SciPy(np.log(C)),'--k')
plt.xlabel('log(n)')
plt.ylabel('log(t)')
plt.title("Temps d'exécution en fonction de la taille n \n en échelle logarithmique de SciPy" )
plt.show()
#plot QRsolve
plt.plot(N,T,'bo',N,fit_fn_QR3(N),'--k')
plt.title("Temps d'exécution en fonction de la taille n de QRsolve" )
plt.xlabel("Taille de la matrice carrée A")
plt.ylabel("Temps d'exécution (sec)")
plt.show()
