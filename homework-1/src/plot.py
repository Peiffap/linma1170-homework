#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:49:50 2018

@author: admin
"""

import matplotlib.pyplot as plt
import numpy as np

def perf_graph():
    R = [2.057, 2.057, 0.077, 6.097, 74.728, 1000]
    U = [0.615, 0.644, 0.037, 2.148, 26.491, 300]
    S = [1.454, 1.428, 0.050, 3.957, 48.314, 600]
    N = [10, 100, 1000, 10000, 100000, 1000000]
    
    logN = np.log(N)
    logR = np.log(R)
    logU = np.log(U)
    logS = np.log(S)
    
    fitR = np.polyfit(logN, logR, 1)
    fitU = np.polyfit(logN, logU, 1)
    fitS = np.polyfit(logN, logS, 1)
    
    fit_fnR = np.poly1d(fitR)
    fit_fnU = np.poly1d(fitU)
    fit_fnS = np.poly1d(fitS)
    
    plt.plot(logN, logR, 'ro', logN, logU, 'bo', logN, logS, 'go', logN, fit_fnR(logN), '--r', logN, fit_fnU(logN), '--b', logN, fit_fnS(logN), '--g')
    plt.xlabel(r"Taille du fichier [MB] en logarithmique")
    plt.ylabel(r"Temps logarithmique [s]")
    plt.title(r"Temps d'exécution en fonction de la taille du fichier d'entrée")
    plt.show()
    

if __name__ == "__main__":
    perf_graph()