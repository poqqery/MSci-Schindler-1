# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 01:31:09 2023

@author: Madhuwrit, Louis
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

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
    
class excitation_solver:
    def __init__(self, interaction_strength, hamiltonian, lattice_constant, N=50):
        """
        Solves the low-temperature excitation spectra for the given
        Hamiltonian—if flat bands exist. Currently only works in 1D; need to
        add functionality for 2 or 3 dimensions—if you're mad.

        Parameters
        ----------
        interaction_strength : float
            Hubbard interaction strength. |U| in literature.
        hamiltonian : function object
            Function taking a wavevector, k, and spin state as input.
            Generally returns a matrix. Assumed to be Hermitian (it damn well
            better be).
        lattice_constant : float
            Lattice spacing from one unit cell to another.
        N : int
            Number of unit cells. This automatically determines the valid
            wavevectors to sample at: the ones that satisfy periodic boundary
            conditions and maintain independence. For the love of God, keep
            this number small if evaluating trions. N must be even!
            Translation of the FBZ does not work correctly if N is not even
            because there will be no periodic state at k = 0. The default
            is 50.

        Returns
        -------
        None.

        """
        
        self._mod_U = interaction_strength
        self._hamiltonian = hamiltonian
        self._a = lattice_constant
        self._N = N
        # Linearly spaced samples of k in the FBZ (currently only 1D)
        # The last k is omitted because it is the same state as the first.
        # Sampled from 0 to 2pi, not -pi to pi to make later operations easier.
        # Use aliasing to translate back to -pi to pi later for plotting.
        self._k_samples = np.linspace(0., 2.*np.pi, self._N + 1)[:-1] / self._a
        
        # Find the flat bands, if any exist
        self._flat_bands_up, self._flat_bands_down = self.identify_flat_bands()
        
    def identify_flat_bands(self):
        """
        Solves for the eigenvalues and eigenvectors of the given Hamiltonian
        at linearly spaced samples of k in the FBZ, determined by the specified
        number of intervals in k-space. It will then try to determine the
        existence of flat bands and return them if present. If no flat bands
        exist, the program terminates here.

        Raises
        ------
        NoFlatBandException()
            Raised if no flat bands exist.

        Returns
        -------
        flat_bands_up : list of int
            List of flat band indices of the Hamiltonian evaluated with spin
            up.
        flat_bands_down : list of int
            List of flat band indices of the Hamiltonian evaluated with spin
            down.

        """
        
        # Number of orbitals/bands = rank of the Hamiltonian matrix
        # So, just evaluate it at k = 0 to see the dimensionality of the matrix
        self._N_orb = self._hamiltonian(0., 1).shape[0]
        # 2 spins, N wavevectors, N_orb eigenvalues
        self._eigenvalues = np.zeros((2, self._N, self._N_orb))
        # 2 spins, N wavevectors, (N_orb x N_orb) eigenvector matrix
        self._eigenvectors = np.zeros((2, self._N, self._N_orb, self._N_orb), dtype="complex128")
        
        # Calculate eigenvalues and eigenvectors of the Hamiltonian
        # First evaluate spin up (0), then spin down (1)
        spins = (0, 1)
        for spin in spins:
            for i in range(self._N):
                sample_hamiltonian = self._hamiltonian(self._k_samples[i], spin)
                sample_eigenvalues, sample_eigenvectors = LA.eigh(sample_hamiltonian)
                #sample_eigenvectors = 0.5*np.array([[1.-np.exp(-1.j * self._k_samples[i]), 0.], [1.+np.exp(-1.j * self._k_samples[i]), 0.]])
                self._eigenvalues[spin,i] += sample_eigenvalues
                self._eigenvectors[spin,i] += sample_eigenvectors
                
        # Now identify flat bands (if any exist).
        # This algorithm is technically flawed because it depends on
        # np.linalg.eigh returning the eigenvalues of the bands in the
        # same "order" consistently, but it'll work for now as long as bands
        # don't cross.
        
        flat_bands_up = []
        flat_bands_down = []
        
        # Test spin up first, then spin down
        for spin in spins:
            for i in range(self._N_orb):
                energies = self._eigenvalues[spin,:,i]
                first_value = energies[0]
                # Check if all the energies are suitably close to the first value
                if np.allclose(first_value, energies, rtol=1e-4, atol=1e-7):
                    # If True, identify as a flat band—otherwise ignore
                    if (spin == 0):
                        flat_bands_up.append(i)
                    else:
                        flat_bands_down.append(i)
            
        return flat_bands_up, flat_bands_down
        
    def trions_3_electrons(self, flat_bands, spins, p):
        
        # Epsilon from the UPC can now be assigned
        self._N_f = len(flat_bands)
        self._epsilon = self._N_f / self._N_orb
        
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,:,flat_bands]
        
        # Define the three spins (0 maps to 1, 1 maps to -1 for up and down respectively)
        # sigma = index 0; sigma prime = index 1; sigma double prime = index 2
        spins = np.array(spins)
        s = -2*spins + 1
        
        # Number of rows and columns in R
        R_dimensionality =  self._N**2 * self._N_f**3
        self._R = np.zeros((3, R_dimensionality, R_dimensionality), dtype="complex128")
        
        # (I honestly can't see any way right now to avoid 10 for loops...)
        # Counts the current row of R being filled in
        R_row_index = 0
        for m_prime in range(self._N_f):
            for n_prime in range(self._N_f):
                for l_prime in range(self._N_f):
                    for k_1_prime in range(self._N):
                        for k_2_prime in range(self._N):
                            # Counts the current column of R being filled in
                            R_column_index = 0
                            for m in range(self._N_f):
                                for n in range(self._N_f):
                                    for l in range(self._N_f):
                                        for k_1 in range(self._N):
                                            for k_2 in range(self._N):
                                                R_sum_0 = 0.j
                                                R_sum_1 = 0.j
                                                R_sum_2 = 0.j
                                                if (((k_1_prime+k_2_prime)%self._N == (k_1+k_2)%self._N) and (m_prime == m)):
                                                    R_sum_0 += np.sum(np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,n_prime]) *\
                                                        reduced_eigenvectors[spins[1],(-k_1)%self._N,:,n] *\
                                                        np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,l_prime]) *\
                                                        reduced_eigenvectors[spins[2],(-k_2)%self._N,:,l]) * s[1] * s[2]
                                                
                                                if ((k_1_prime == k_1) and (n_prime == n)):
                                                    R_sum_1 += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1+k_2_prime)%self._N,:,m_prime]) *\
                                                        reduced_eigenvectors[spins[0],(p+k_1+k_2)%self._N,:,m] *\
                                                        np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,l_prime]) *\
                                                        reduced_eigenvectors[spins[2],(-k_2)%self._N,:,l]) * s[0] * s[2]
                                                        
                                                if ((k_2_prime == k_2) and (l_prime == l)):
                                                    R_sum_2 += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_prime+k_2)%self._N,:,m_prime]) *\
                                                        reduced_eigenvectors[spins[0],(p+k_1+k_2)%self._N,:,m] *\
                                                        np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,n_prime]) *\
                                                        reduced_eigenvectors[spins[1],(-k_1)%self._N,:,n]) * s[0] * s[1]
                                                        
                                                self._R[:,R_row_index,R_column_index] += np.array([R_sum_0, R_sum_1, R_sum_2]) / self._N
                                                
                                                R_column_index += 1
                            
                            R_row_index += 1
                            
        A = self._R[0]
        B = self._R[1]
        C = self._R[2]
        A_norm = LA.norm(A, ord="fro")
        B_norm = LA.norm(B, ord="fro")
        C_norm = LA.norm(C, ord="fro")
        
        commutator_AB = np.abs(A @ B - B @ A)
        commutator_AC = np.abs(A @ C - C @ A)
        commutator_BC = np.abs(B @ C - C @ B)
        max_AB = np.max(commutator_AB)
        mean_AB = np.mean(commutator_AB)
        max_AC = np.max(commutator_AC)
        mean_AC = np.mean(commutator_AC)
        max_BC = np.max(commutator_BC)
        mean_BC = np.mean(commutator_BC)
        
        A /= A_norm
        B /= B_norm
        C /= C_norm
        
        commutator_AB = np.abs(A @ B - B @ A)
        commutator_AC = np.abs(A @ C - C @ A)
        commutator_BC = np.abs(B @ C - C @ B)
        norm_max_AB = np.max(commutator_AB)
        norm_mean_AB = np.mean(commutator_AB)
        norm_max_AC = np.max(commutator_AC)
        norm_mean_AC = np.mean(commutator_AC)
        norm_max_BC = np.max(commutator_BC)
        norm_mean_BC = np.mean(commutator_BC)
        
        return np.array([[max_AB, mean_AB, norm_max_AB, norm_mean_AB],\
                         [max_AC, mean_AC, norm_max_AC, norm_mean_AC],\
                         [max_BC, mean_BC, norm_max_BC, norm_mean_BC]])
            
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
            
#%%

N = np.arange(4, 19, 2)

AB = np.zeros((len(N), 4))
AC = np.zeros((len(N), 4))
BC = np.zeros((len(N), 4))

for i, N_value in enumerate(N):
    solver = excitation_solver(1., dimerized_SSH, 1., N_value)
    values = np.zeros((N_value, 3, 4))
    for p_value in range(N_value):
        values[p_value] += solver.trions_3_electrons([0], [0, 0, 0], p_value)
        
    values = np.mean(values, axis=0)
    AB[i] += values[0]
    AC[i] += values[1]
    BC[i] += values[2]
    
#%%

coeffs, cov = np.polyfit(np.log(N)[2:], np.log(AB[:,0])[2:], deg=1, cov=True)
print(coeffs)

plt.plot(N, AB[:,0], "x", ms="10", color="blue", label=r"Mean of $|[A, B]|$")
plt.plot(N, np.exp(coeffs[1]) * N**coeffs[0], "--", color="red")
plt.xlabel(r"log$_{2}(N)$")
plt.ylabel(r"log$_{2}(Value)$")
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.legend()
plt.grid(True)

plt.show()