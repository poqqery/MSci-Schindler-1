# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:06:45 2024

@author: Madhuwrit and Louis
"""

import numpy as np
from solver import excitation_solver
from copy import deepcopy

#%%

sigma_1 = np.array([[0., 1.], [1., 0.]], dtype="complex128")
sigma_2 = np.array([[0., -1.j], [1.j, 0.]])
sigma_3 = np.array([[1., 0.], [0., -1.]], dtype="complex128")

def dimerized_SSH(k, spin):
    """
    Hamiltonian of the dimerized SSH chain (taken from the paper).

    Parameters
    ----------
    k : float
        Wavenumber to sample at.
    spin : int
        Spin state (0 for up, 1 for down). Doesn't make any difference in this
        Hamiltonian; exists for consistency with Hamiltonians that do.

    Returns
    -------
    h : ndarray of complex128
        Hamiltonian matrix at the given k with the given spin.

    """
    
    h = np.sin(k) * sigma_2 + np.cos(k) * sigma_3
    
    return h

N = 8

test = excitation_solver(1., dimerized_SSH, 1., N)
U = test._eigenvectors[:,:,:,0]

#%%

def V(U):
    matrix = np.zeros((N**2, N*2), dtype="complex128")
    for k_1 in range(N):
        for k_2 in range(N):
            for q in range(N):
                for a in range(2):
                    if ((k_1 + k_2)%N == q):
                        matrix[N*k_1 + k_2, 2*q + a] += np.conj(U[0,(-k_1)%N,a] * U[0,(-k_2)%N,a])
                        
    return matrix / np.sqrt(N)

test.trions_3_electrons([0], [0, 0, 0])
A = test._R[0]

V_A = V(U)

reduced_A = np.conj(V_A.T) @ V_A
epsilon, v = np.linalg.eigh(reduced_A)

A_evec_0 = V_A @ v[:,1]
print(A_evec_0 / (A @ A_evec_0))

#%%

def A_evec(j,U):
    evec = np.zeros(2*N, dtype="complex128")
    sign = np.array([1,-1])
    for k in range(N):
        for mu in range(2):
            evec[2*k + mu] += np.conj(U[0,(-j)%N,0] * U[0,(-k)%N,0]) + sign[mu]*np.conj(U[0,(-j)%N,1] * U[0,(-k)%N,1])
                
    return evec

eigenvector = A_evec(1, U)

print(eigenvector / v[:,1])