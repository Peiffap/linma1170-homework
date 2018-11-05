# The function mysolve(A, b) is invoked by ccore.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ccore.py -clscale float
# where the line argument 'clscale' allows the global coarsening of the mesh
# -clscale 1  fine mesh
# -clscale 10 coarse mesh


# Homework 2 : SVD and condition number
# @author Gilles Peiffer, 24321600
# gilles.peiffer@student.uclouvain.be
# Nov. 9, 2018


SolverType = 'scipy'

import numpy as np
import scipy
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import csv # for analysis

def mysolve(A, b, clscale):
    if SolverType == 'scipy':
        cond_anal(A, clscale)
        return True, scipy.sparse.linalg.spsolve(A, b)  # Needs import
    #elif SolverType == 'QR':
        # return True, QRsolve(A, b)
    #elif SolverType == 'LU':
        # write here your code for the LU solver
    #elif SolverType == 'GMRES':
        # write here your code for the LU solver
    else:
        return False, 0
     

def cond_anal(A, clscale):
    if clscale == 1:
        wmode = 'w'
    else:
        wmode = 'a'
        
    with open('cond_anal_clscale.csv', mode=wmode) as dat:
        dat_writer = csv.writer(dat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dat_writer.writerow([clscale, np.linalg.cond(A)])


def low_rank_approx(A,b):
    U, s, Vh = scipy.linalg.svd(A)
    ensum = 0
    for i in range(len(s)):
        ensum += np.dot(b, (np.outer(U.T[i], Vh[i])/s[i]))
    return ensum


def cond_plot():
    """
    This function plots graphs that make it possible to easily analyse
    the impact of different variations to the model of the coil in ccore.py.
    Data is written from the file written in the previous functions
    
    
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
    """
    
    with open('cond_anal_clscale.csv') as clres:
        clres_reader = csv.reader(clres, delimiter=',')
        clscale = []
        cond = []
        for row in clres_reader:
            clscale.append(row[0])
            cond.append(row[1])
            
    

if __name__ == "__main__":
    cond_plot()