# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 01:29:11 2023

@author: Madhuwrit, Louis
"""

import numpy as np
import matplotlib.pyplot as plt
from solver_multiband import excitation_solver
from copy import deepcopy

def update_params_sim():
    params = {
        "axes.labelsize": 18,
        "font.size": 18,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.figsize": [10, 6],
        "figure.autolayout": True
    }
    plt.rcParams.update(params)
    
update_params_sim()

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

def SSH(k, spin):
    
    return -1. * np.sin(k) * sigma_3 + np.cos(k) * sigma_1

def SSH_triplet(k, spin):
    
    return np.array([[np.cos(k), (1+np.exp(1.j*k))/np.sqrt(2), -1.j*np.sin(k)],\
                     [(1.+np.exp(-1.j*k))/np.sqrt(2), 0., (-1.+np.exp(-1.j*k))/np.sqrt(2)],\
                     [1.j*np.sin(k), (-1.+np.exp(1.j*k))/np.sqrt(2), -np.cos(k)]], dtype="complex128")

N = 16

test = excitation_solver(1., SSH_triplet, 1., N)

#%% Overlaying the minimum energies of two independent, spin-zero Cooper pairs
# onto a 4-particle spin-zero spectrum

test.charge_2_spectra([0], plot=False)
test.charge_4_electrons([0], [0, 0, 1, 1], plot=False)

charge_2_energies = deepcopy(test._charge_2_energies)
charge_4_energies = deepcopy(test._four_electrons_energies)
rolled_k = np.roll(np.linspace(-np.pi, np.pi, N+1)[:-1], -(N // 2))

lowest_energies = np.ones(N) * 10.

for i in range(N):
    for j in range(N):
        lowest_energy = np.min(charge_2_energies[i]) + np.min(charge_2_energies[j])
        p = (i + j) % N
        
        if (lowest_energy < lowest_energies[p]):
            lowest_energies[p] = lowest_energy * 1.
            
            
for i, k_value in enumerate(rolled_k):
    length = len(charge_4_energies[i])
    plt.plot(np.ones(length)*k_value, charge_4_energies[i], ".", color="green")

plt.plot(rolled_k, lowest_energies, ".", color="blue")
plt.plot(np.array([-np.pi, np.pi]), 2.*1.*0.5*np.ones(2), "--", color="black", label=r"$2 \epsilon |U|$")
    
plt.grid(True)
plt.legend()
plt.title("Excitation Spectrum", fontsize=18)
plt.xlabel(r"$pa$")
plt.ylabel("Energy")
plt.show()