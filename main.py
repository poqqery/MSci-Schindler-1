# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 01:29:11 2023

@author: Madhuwrit, Louis
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import excitation_solver

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
        Matrix of the Hamiltonian at the given k with the given spin.

    """
    
    h = np.sin(k) * sigma_2 + np.cos(k) * sigma_3
    
    return h

def zig_zag(k, spin):
    #assume hopping term is 1
    sigma_1=np.array([[0. , 1.],[1. , 0.]])
    sigma_2 = np.array([[0., -1.j], [1.j, 0.]])
    
    h=np.sqrt(2)*np.cos(k)*np.array([[ 1. , 0.],[ 0., 0.]])+(1+np.cos(k))*sigma_1+np.sin(k)*sigma_2
    
    return h

test = excitation_solver(1., dimerized_SSH, 1., 30)