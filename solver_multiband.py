# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 01:31:09 2023

@author: Madhuwrit, Louis
"""

import numpy as np
import numpy.linalg as LA
from scipy import sparse
import matplotlib.pyplot as plt
from multiprocessing import Process

class NoFlatBandException(Exception):
    "Raised in the absence of flat bands."
    pass

class NonDegenerateFlatBandsException(Exception):
    "Raised if chosen flat bands to analyze excitation spectra of are not degenerate."
    pass

class UPCFailedException(Exception):
    "Raised if the given model does not obey the uniform pairing condition (UPC)."
    
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
        self._true_indices = None
        
    def check_band_degeneracy(self, flat_bands):
        """
        Checks that the specified bands to calculate excitation spectra with
        are degenerate amongst each other. Terminates if the bands are not
        degenerate.

        Parameters
        ----------
        flat_bands : list of ints
            Indices of bands to analyze.

        Raises
        ------
        NonDegenerateFlatBandsException
            Raised if the given bands are not degenerate.

        Returns
        -------
        None.

        """
        
        # First check that the chosen flat bands to analyze do actually
        # all lie at the same energy
        first_energies_up = self._eigenvalues[0,0,flat_bands]
        first_energies_down = self._eigenvalues[1,0,flat_bands]
        first_energies = np.concatenate((first_energies_up, first_energies_down))
        if np.allclose(first_energies[0], first_energies, rtol=1e-4, atol=1e-7):
            # Continue; the bands are degenerate amongst each other
            pass
        
        else:
            # Terminate otherwise
            raise NonDegenerateFlatBandsException
            
    def check_UPC(self, reduced_eigenvectors):
        """
        Checks that the given model satisfies the uniform pairing condition
        (UPC). Terminates if it does not.

        Parameters
        ----------
        reduced_eigenvectors : ndarray of complex128
            Eigenvectors associated with the given flat bands to analyze.

        Raises
        ------
        UPCFailedException
            Raised if the model does not satisfy the UPC.

        Returns
        -------
        None.

        """
        
        p_up = np.einsum("ijk,ilk->jl", reduced_eigenvectors[0], np.conjugate(reduced_eigenvectors[0])) / self._N
        p_down = np.einsum("ijk,ilk->jl", reduced_eigenvectors[1], np.conjugate(reduced_eigenvectors[1])) / self._N
        
        if np.allclose(self._epsilon, np.diag(p_up), rtol=1e-4, atol=1e-7):
            # Continue; the model satisfies the UPC and is valid for analysis
            pass
        
        else:
            # Terminate otherwise
            raise UPCFailedException
            
        if np.allclose(self._epsilon, np.diag(p_down), rtol=1e-4, atol=1e-7):
            pass
        
        else:
            raise UPCFailedException
        
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
        unitary = 2.**(-0.5) * np.array([[1., 0., 1.], [0., np.sqrt(2.), 0.], [-1., 0., 1.]])
        spins = (0, 1)
        for spin in spins:
            for i in range(self._N):
                sample_hamiltonian = self._hamiltonian(self._k_samples[i], spin)
                sample_eigenvalues, sample_eigenvectors = LA.eigh(sample_hamiltonian)
                #sample_eigenvectors = 0.5*np.array([[1.-np.exp(-1.j * self._k_samples[i]), 0.], [1.+np.exp(-1.j * self._k_samples[i]), 0.]])
                sample_eigenvectors = unitary @ np.array([np.array([1., -1., 0.])/np.sqrt(2.),\
                                                         np.array([1., 1., -2.*np.exp(1.j*self._k_samples[i])])/np.sqrt(6.),\
                                                         np.array([1., 1., np.exp(1.j*self._k_samples[i])])/np.sqrt(3.)]).T
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
                        
        if ((len(flat_bands_up) + len(flat_bands_down)) != 0):
            print("%.i flat bands detected (spin up):" % len(flat_bands_up))
            print(flat_bands_up)
            print("%.i flat bands detected (spin down):" % len(flat_bands_down))
            print(flat_bands_up)
            
        else:
            # Terminate otherwise—there's no point continuing if there are
            # no flat bands.
            raise NoFlatBandException
            
        # To-do: implement an algorithm that correctly keeps a given band
        # in a given column even if there are intersecting bands.
            
        return flat_bands_up, flat_bands_down
    
    def charge_1_spectra(self, flat_bands, spin, plot=True):
        """
        Calculates the charge +1 excitation spectra for the specified
        degenerate flat bands. Everything should give a flat spectrum at a
        single energy regardless of the given Hamiltonian. Terminates if
        the given bands are not degenerate.

        Parameters
        ----------
        flat_bands : list of ints
            Indices of the flat bands to analyze.
        spin : int
            Electron/hole spin. 0 for up, 1 for down.

        Returns
        -------
        None.

        """
        
        # Check the bands all lie at the same energy. Terminate otherwise.
        self.check_band_degeneracy(flat_bands)
        # Epsilon from the UPC can now be assigned
        self._N_f = len(flat_bands)
        self._epsilon = self._N_f / self._N_orb
        
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,:,flat_bands]
        # Check the model obeys the UPC. Terminate otherwise.
        self.check_UPC(reduced_eigenvectors)
        
        charge_1_energies = []
        
        for i in range(self._N):
            # Find the appropriate matrix of eigenvectors for the given k
            U = reduced_eigenvectors[spin,i]
            # Calculate the matrix, R
            R = 0.5 * self._mod_U * self._epsilon * np.einsum("am,an->mn", np.conjugate(U), U)
            # Now get the eigenenergies
            charge_1_energies.append(LA.eigvalsh(R))
            
        self._charge_1_energies = np.array(charge_1_energies)
        
        if (plot == True):
            scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
            # Plot the excitation spectra (should always be flat)
            for i in range(self._charge_1_energies.shape[1]):
                plt.plot(scaled_k, np.roll(self._charge_1_energies[:,i], self._N // 2), ".")
                
            plt.grid(True)
            plt.title("Excitation Spectrum", fontsize=18)
            plt.xlabel(r"$ka$")
            plt.ylabel("Energy")
            plt.show()
        
    def charge_2_spectra(self, flat_bands, plot=True):
        """
        Calculates the Cooper pair excitation spectra for the specified
        degenerate flat bands. Automatically assumes opposite spins for the
        electrons since the alternatives are trivial. Terminates if the given
        bands are not degenerate. Assumes spin-0 Cooper pairs.

        Parameters
        ----------
        flat_bands : list of ints
            Indices of the flat bands to analyze.

        Returns
        -------
        None.

        """
        
        # Check the bands all lie at the same energy. Terminate otherwise.
        self.check_band_degeneracy(flat_bands)
        # Epsilon from the UPC can now be assigned
        self._N_f = len(flat_bands)
        self._epsilon = self._N_f / self._N_orb
        
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,:,flat_bands]
        # Check the model obeys the UPC. Terminate otherwise.
        self.check_UPC(reduced_eigenvectors)
        
        charge_2_energies = []
        
        # To solve at each pair momentum, p, all the FBZ momenta, k, need to be
        # summed over (it's a whole lot worse with trions...)
        
        # Iterate over each p
        for i in range(self._N):
            h = 0.j
            # Iterate over each k
            for j in range(self._N):
                # See page 30 of the paper for clarification on the process being
                # performed here
                U_k = reduced_eigenvectors[0,j]
                P_k = U_k @ np.conjugate(U_k.T)
                # The modulo wraps it back into an FBZ wavevector (whose related
                # eigenvectors have already been calculated)
                U_p_plus_k = reduced_eigenvectors[0,(i + j) % self._N]
                P_p_plus_k = U_p_plus_k @ np.conjugate(U_p_plus_k.T)
                h += P_p_plus_k * P_k.T
                
            energies = LA.eigvalsh(h / self._N)
            charge_2_energies.append(self._mod_U * (self._epsilon - energies))
            
        self._charge_2_energies = np.array(charge_2_energies)
        
        if (plot == True):
            for i in range(self._charge_2_energies.shape[1]):
                plt.plot(self._k_samples * self._a, self._charge_2_energies[:,i], ".", color="blue")
                
            plt.plot(np.array([0., 2.*np.pi]), self._mod_U * self._epsilon * np.ones(2), "--", color="black", label=r"$\epsilon |U|$")
            
            plt.grid(True)
            plt.legend()
            plt.title("Cooper Pair Excitation Spectrum", fontsize=18)
            plt.xlabel(r"$pa$")
            plt.ylabel("Energy")
            plt.show()
        
    def charge_2_spectra_full(self, flat_bands, spins, plot=True):
        
        # Check the bands all lie at the same energy. Terminate otherwise.
        self.check_band_degeneracy(flat_bands)
        # Epsilon from the UPC can now be assigned
        self._N_f = len(flat_bands)
        self._epsilon = self._N_f / self._N_orb
        
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,:,flat_bands]
        # Check the model obeys the UPC. Terminate otherwise.
        self.check_UPC(reduced_eigenvectors)
        
        # Define the three spins (0 maps to 1, 1 maps to -1 for up and down respectively)
        # sigma = index 0; sigma prime = index 1; sigma double prime = index 2
        spins = np.array(spins)
        s = -2*np.array(spins) + 1
        
        # Number of rows and columns in R
        R_dimensionality =  self._N * self._N_f**2
        self._R = np.zeros((self._N, R_dimensionality, R_dimensionality), dtype="complex128")
        
        # Calculate a component of R for all p at once
        p = np.arange(self._N)
        
        R_row_index = 0
        for m_prime in range(self._N_f):
            for n_prime in range(self._N_f):
                for k_prime in range(self._N):
                    R_col_index = 0
                    for m in range(self._N_f):
                        for n in range(self._N_f):
                            for k in range(self._N):
                                R_sum = np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_prime)%self._N,:,m_prime]) *\
                                    reduced_eigenvectors[spins[0],(p+k)%self._N,:,m] *\
                                    np.conj(reduced_eigenvectors[spins[1],(-k_prime)%self._N,:,n_prime]) *\
                                    reduced_eigenvectors[spins[1],(-k)%self._N,:,n], axis=1) * s[0] * s[1]
                                    
                                self._R[:,R_row_index,R_col_index] += R_sum / self._N
                                
                                R_col_index += 1
                    
                    R_row_index += 1
        
        # Add the identity matrix part to everything at once
        self._R[:,np.arange(R_dimensionality),np.arange(R_dimensionality)] += self._epsilon
                    
        
        # Now apply the correction (the second R term)
        # Applies only if the spins are the same
        if (spins[0] == spins[1]):
            m = np.concatenate([i*np.ones(self._N*self._N_f) for i in range(self._N_f)]).astype("int32")
            n = np.concatenate([i*np.ones(self._N) for i in range(self._N_f)] * self._N_f).astype("int32")
            q = np.array([i for i in range(self._N)] * self._N_f**2).astype("int32")
            for p_value in range(self._N):
                rearranged = n*(self._N * self._N_f) + m*self._N + (-p_value-q) % self._N
                self._R[p_value] -= self._R[p_value,rearranged]
                
        # Convert 3-dimensional array, self._R, to a list of matrices at the different p
        self._R = list(self._R)
                
        # Remove duplicate (non-independent) states and those forbidden by the PEP
        for p_value in range(self._N):
            independent_states = []
            allowed_indices = []
            # Disallowed states occur only when the spins are the same
            if (spins[0] == spins[1]):
                for m_value in range(self._N_f):
                    for n_value in range(self._N_f):
                        for k_value in range(self._N):
                            # Go through each state at this p
                            # Sorting removes dependence on the operator order of gamma dagger
                            momentum_state = np.array([(p_value + k_value) % self._N, (-k_value) % self._N])
                            sorted_indices = np.argsort(momentum_state)
                            momentum_state = momentum_state[sorted_indices]
                            band_state = np.array([m_value, n_value])[sorted_indices]
                            
                            unique_momenta = np.unique(momentum_state)
                            # Everything is automatically sorted if all momenta are different;
                            # check otherwise that bands are also sorted
                            if (len(unique_momenta) != 2):
                                current = 0
                                for momentum in unique_momenta:
                                    count = sum(momentum_state == momentum)
                                    band_state[current:current+count] = np.sort(band_state[current:current+count])
                                    current += count
                                    
                            state = list(momentum_state) + list(band_state)
                            
                            if state in independent_states:
                                # This state is not independent (it has already been counted)
                                independent = False
                            else:
                                independent_states.append(state)
                                independent = True
            
                            # Now check if the state is forbidden by the PEP (momenta are the same)
                            if (((p_value + k_value) % self._N == (-k_value) % self._N) and (m_value == n_value)):
                                allowed = False
                            else:
                                allowed = True
                                
                            # If the state is independent and allowed, it is kept in the matrix
                            if ((independent == True) and (allowed == True)):
                                allowed_indices.append(True)
                            else:
                                allowed_indices.append(False)
            
            # If the spins are different, states at all k are allowed
            else:
                allowed_indices = np.array([True] * R_dimensionality)
                    
            # Project down the matrix to the independent, allowed states
            # Filter the rows, then the columns (doesn't matter which way)
            self._R[p_value] = self._R[p_value][allowed_indices][:,allowed_indices]
                    
        self._cooper_pair_energies = []
        
        # Calculate the energy spectrum
        for p_value in range(self._N):
            energies = LA.eigvalsh(self._R[p_value])
            self._cooper_pair_energies.append(self._mod_U * energies)
            
        if (plot == True):
            scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
            scaled_k = np.roll(scaled_k, -(self._N // 2))
            for i, k_value in enumerate(scaled_k):
                length = len(self._cooper_pair_energies[i])
                plt.plot(np.ones(length)*k_value, self._cooper_pair_energies[i], ".", color="blue")
            
            plt.plot(np.array([-np.pi, np.pi]), self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$\epsilon |U|$")
                
            plt.grid(True)
            plt.legend()
            plt.title("Cooper Pair Excitation Spectrum", fontsize=18)
            plt.xlabel(r"$pa$")
            plt.ylabel("Energy")
            plt.show()
        
    def charge_2_mixed(self, flat_bands, spins, plot=True):
        """
        Calculates the electron-hole pair excitation spectra for the specified
        degenerate flat bands. Terminates if the given bands are not
        degenerate.

        Parameters
        ----------
        flat_bands : list of ints
            Indices of the flat bands to analyze.
        spins : array or similar of ints
            Spins of the two fermions; left is the electron and right the hole.
            0 for up and 1 for down.

        Returns
        -------
        None.

        """
        
        # Check the bands all lie at the same energy. Terminate otherwise.
        self.check_band_degeneracy(flat_bands)
        # Epsilon from the UPC can now be assigned
        self._N_f = len(flat_bands)
        self._epsilon = self._N_f / self._N_orb
        
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,:,flat_bands]
        # Check the model obeys the UPC. Terminate otherwise.
        self.check_UPC(reduced_eigenvectors)
        
        # Define the two spins (0 maps to 1, 1 maps to -1 for up and down respectively)
        # sigma = index 0; sigma prime = index 1
        s = -2*np.array(spins) + 1
        spin_product = s[0] * s[1]
        
        charge_2_mixed_energies = []
        
        # Iterate over each p
        for p in range(self._N):
            h = 0.j
            # Iterate over each k
            for k in range(self._N):
                U_minus_p_minus_k = reduced_eigenvectors[spins[0],(-p-k)%self._N]
                P_minus_p_minus_k = U_minus_p_minus_k @ np.conj(U_minus_p_minus_k.T)
                U_minus_k = reduced_eigenvectors[spins[1],(-k)%self._N]
                P_minus_k = U_minus_k @ np.conj(U_minus_k.T)
                h += np.conj(P_minus_p_minus_k) * P_minus_k
                
            energies = LA.eigvalsh(h / self._N)
            charge_2_mixed_energies.append(self._mod_U * (self._epsilon - spin_product * energies))
            
        self._charge_2_mixed_energies = np.array(charge_2_mixed_energies)
            
        if (plot == True):
            scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
            # Plot the excitation spectra
            for i in range(self._charge_2_mixed_energies.shape[1]):
                plt.plot(scaled_k, np.roll(self._charge_2_mixed_energies[:,i], self._N // 2), ".", color="blue")
                
            plt.plot(np.array([-np.pi, np.pi]), self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$\epsilon |U|$")
                
            plt.grid(True)
            plt.legend()
            plt.title("Charge +2 (Mixed) Excitation Spectrum", fontsize=18)
            plt.xlabel(r"$pa$")
            plt.ylabel("Energy")
            plt.show()
        
    def trions_3_electrons_multiband(self, flat_bands, spins, num_eigenvalues=10, plot=True):
        
        # Check the bands all lie at the same energy. Terminate otherwise.
        self.check_band_degeneracy(flat_bands)
        # Epsilon from the UPC can now be assigned
        self._N_f = len(flat_bands)
        self._epsilon = self._N_f / self._N_orb
        
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,:,flat_bands]
        # Check the model obeys the UPC. Terminate otherwise.
        self.check_UPC(reduced_eigenvectors)
        
        # Define the three spins (0 maps to 1, 1 maps to -1 for up and down respectively)
        spins = np.array(spins)
        s = -2*spins + 1
        
        # Number of rows and columns in R
        R_dimensionality =  self._N**2 * self._N_f**3
        const_m = self._N**2 * self._N_f**2
        const_n = self._N**2 * self._N_f
        const_l = self._N**2
        
        m = np.concatenate([i*np.ones(const_m, dtype="int32") for i in range(self._N_f)]).astype("int32")
        n = np.concatenate([i*np.ones(const_n, dtype="int32") for i in range(self._N_f)] * self._N_f).astype("int32")
        l = np.concatenate([i*np.ones(const_l, dtype="int32") for i in range(self._N_f)] * self._N_f**2).astype("int32")
        q_1 = np.concatenate([i*np.ones(self._N) for i in range(self._N)] * self._N_f**3).astype("int32")
        q_2 = np.concatenate([np.arange(self._N)] * self._N * self._N_f**3).astype("int32")
        
        self._energies = []
        
        # Sparse matrices; do one p at a time to prevent RAM destruction
        for p in range(self._N):
            row_indices = []
            col_indices = []
            values = []
            for row in range(R_dimensionality):
                # Only fill the diagonal and upper-triangular half, then use
                # Hermitianity to fill the lower triangular half
                # Find row state
                m_row = m[row]
                n_row = n[row]
                l_row = l[row]
                k_1_row = q_1[row]
                k_2_row = q_2[row]
                for col in range(row, R_dimensionality):
                    # Find column state
                    m_col = m[col]
                    n_col = n[col]
                    l_col = l[col]
                    k_1_col = q_1[col]
                    k_2_col = q_2[col]
                    
                    R_sum = 0.j
                    if (((k_1_row+k_2_row)%self._N == (k_1_col+k_2_col)%self._N) and (m_row == m_col)):
                        R_sum += np.sum(np.conj(reduced_eigenvectors[spins[1],(-k_1_row)%self._N,:,n_row]) *\
                            reduced_eigenvectors[spins[1],(-k_1_col)%self._N,:,n_col] *\
                            np.conj(reduced_eigenvectors[spins[2],(-k_2_row)%self._N,:,l_row]) *\
                            reduced_eigenvectors[spins[2],(-k_2_col)%self._N,:,l_col]) * s[1] * s[2]
                    
                    if ((k_1_row == k_1_col) and (n_row == n_col)):
                        R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_col+k_2_row)%self._N,:,m_row]) *\
                            reduced_eigenvectors[spins[0],(p+k_1_col+k_2_col)%self._N,:,m_col] *\
                            np.conj(reduced_eigenvectors[spins[2],(-k_2_row)%self._N,:,l_row]) *\
                            reduced_eigenvectors[spins[2],(-k_2_col)%self._N,:,l_col]) * s[0] * s[2]
                            
                    if ((k_2_row == k_2_col) and (l_row == l_col)):
                        R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_row+k_2_col)%self._N,:,m_row]) *\
                            reduced_eigenvectors[spins[0],(p+k_1_col+k_2_col)%self._N,:,m_col] *\
                            np.conj(reduced_eigenvectors[spins[1],(-k_1_row)%self._N,:,n_row]) *\
                            reduced_eigenvectors[spins[1],(-k_1_col)%self._N,:,n_col]) * s[0] * s[1]
                            
                    if (R_sum != 0.j):
                        # Specify an explicit element only if non-zero
                        if (row != col):
                            # Fill in both halves
                            row_indices.extend([row, col])
                            col_indices.extend([col, row])
                            values.extend([R_sum / self._N, np.conjugate(R_sum / self._N)])
                        else:
                            # Fill in diagonal
                            row_indices.append(row)
                            col_indices.append(col)
                            values.append(R_sum / self._N)
                            
                    if (row == col):
                        # Add identity matrix
                        row_indices.append(row)
                        col_indices.append(col)
                        values.append(1.5 * self._epsilon)
                        
            # Construct R(p) as a sparse matrix
            R = sparse.csr_array((values, (row_indices, col_indices)), shape=(R_dimensionality, R_dimensionality))
            
            # Apply corrections for the PEP
            q_rearranged = (-p-q_1-q_2) % self._N
            R_copy = R.copy()
            if (spins[0] == spins[1]):
                rearranged_rows = n*const_m + m*const_n + l*const_l + q_rearranged*self._N + q_2
                R -= R_copy[rearranged_rows,:]
                
            if (spins[0] == spins[2]):
                rearranged_rows = l*const_m + n*const_n + m*const_l + q_1*self._N + q_rearranged
                R -= R_copy[rearranged_rows,:]
                
            if (spins[1] == spins[2]):
                rearranged_rows = m*const_m + l*const_n + n*const_l + q_2*self._N + q_1
                R -= R_copy[rearranged_rows,:]
                
            if (spins[0] == spins[1] == spins[2]):
                rearranged_rows = l*const_m + m*const_n + n*const_l + q_rearranged*self._N + q_1
                R += R_copy[rearranged_rows,:]
                rearranged_rows = n*const_m + l*const_n + m*const_l + q_2*self._N + q_rearranged
                R += R_copy[rearranged_rows,:]
                
            del R_copy
            
            # Remove duplicate (non-independent) states and those forbidden by
            # the PEP
            independent_states = []
            allowed_indices = []
            index = 0
            for m_value in range(self._N_f):
                for n_value in range(self._N_f):
                    for l_value in range(self._N_f):
                        for k_1 in range(self._N):
                            for k_2 in range(self._N):
                                # Sorting removes (some) dependence on operator order
                                momentum_state = np.array([(p + k_1 + k_2)%self._N, (-k_1)%self._N, (-k_2)%self._N])
                                sorted_indices = momentum_state.argsort()
                                momentum_state = momentum_state[sorted_indices]
                                band_state = np.array([m_value, n_value, l_value])[sorted_indices]
                                spin_state = spins[sorted_indices]
                                    
                                unique_momenta = np.unique(momentum_state)
                                # Everything is automatically sorted if all momenta are different;
                                # check otherwise that spins are also sorted
                                if (len(unique_momenta) != 3):
                                    current = 0
                                    for momentum in unique_momenta:
                                        count = sum(momentum_state == momentum)
                                        spin_state[current:current+count] = np.sort(spin_state[current:current+count])
                                        band_state[current:current+count] = np.sort(band_state[current:current+count])
                                        current += count
                                
                                state = np.vstack((momentum_state, band_state, spin_state)).T
                                list_state = state.tolist()
                                if list_state in independent_states:
                                    # This state is not independent (it has already been counted)
                                    independent = False
                                else:
                                    independent_states.append(list_state)
                                    independent = True
                                    
                                # Check if the state is forbidden by the PEP (no need to check
                                # if the state is already non-independent)
                                if (independent == True):
                                    if (np.array_equal(state[0], state[1]) or np.array_equal(state[1], state[2]) or np.array_equal(state[0], state[2])):
                                        # Disallowed by the PEP if True
                                        allowed = False
                                    else:
                                        allowed = True
                                else:
                                    allowed = False
                                    
                                if ((independent == True) and (allowed == True)):
                                    allowed_indices.append(True)
                                else:
                                    allowed_indices.append(False)
                                    
                                index += 1
                                
            # Project down R
            R = R[allowed_indices,:][:,allowed_indices]
            energies = self._mod_U * sparse.linalg.eigsh(R, k=num_eigenvalues, which="SA", return_eigenvectors=False)
            #energies = self._mod_U * np.linalg.eigvalsh(R)
            self._energies.append(energies)
            
        if (plot == True):
            for i in range(len(self._k_samples)):
                length = len(self._energies[i])
                plt.plot(np.ones(length)*self._k_samples[i] * self._a, self._energies[i], ".", color="blue")
                
            plt.plot(np.array([0., 2.*np.pi]), 1.5 * self._mod_U * self._epsilon * np.ones(2), "--", color="black", label=r"$\frac{3\epsilon |U|}{2}$")
            
            plt.grid(True)
            plt.legend()
            plt.title("Trion (Electrons) Excitation Spectrum", fontsize=18)
            plt.xlabel(r"$pa$")
            plt.ylabel("Energy")
            plt.show()
        
    def trions_2_electrons_1_hole(self, flat_bands, spins, plot=True):
        
        # Check the bands all lie at the same energy. Terminate otherwise.
        self.check_band_degeneracy(flat_bands)
        # Epsilon from the UPC can now be assigned
        self._N_f = len(flat_bands)
        self._epsilon = self._N_f / self._N_orb
        
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,:,flat_bands]
        # Check the model obeys the UPC. Terminate otherwise.
        self.check_UPC(reduced_eigenvectors)
        
        # Define the three spins (0 maps to 1, 1 maps to -1 for up and down respectively)
        # sigma = index 0; sigma prime = index 1; sigma double prime = index 2
        spins = np.array(spins)
        s = -2*spins + 1
        
        # Number of rows and columns in R
        R_dimensionality =  self._N**2 * self._N_f**3
        self._R = np.zeros((self._N, R_dimensionality, R_dimensionality), dtype="complex128")
        
        # Calculate a component of R for all p at once
        p = np.arange(self._N)
        
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
                                                R_sum = 0.j
                                                if ((k_1_prime == k_1) and (m_prime == m)):
                                                    R_sum -= np.sum(np.conj(reduced_eigenvectors[spins[1],(-k_2_prime)%self._N,:,n_prime]) *\
                                                        reduced_eigenvectors[spins[1],(-k_2)%self._N,:,n] *\
                                                        reduced_eigenvectors[spins[2],(-p-k_1-k_2_prime)%self._N,:,l_prime] *\
                                                        np.conj(reduced_eigenvectors[spins[2],(-p-k_1-k_2)%self._N,:,l]), axis=1) * s[1] * s[2]
                                                
                                                if ((k_2_prime == k_2) and (n_prime == n)):
                                                    R_sum -= np.sum(np.conj(reduced_eigenvectors[spins[0],(-k_1_prime)%self._N,:,m_prime]) *\
                                                        reduced_eigenvectors[spins[0],(-k_1)%self._N,:,m] *\
                                                        reduced_eigenvectors[spins[2],(-p-k_1_prime-k_2)%self._N,:,l_prime] *\
                                                        np.conj(reduced_eigenvectors[spins[2],(-p-k_1-k_2)%self._N,:,l]), axis=1) * s[0] * s[2]
                                                        
                                                if (((k_1_prime + k_2_prime)%self._N == (k_1 + k_2)%self._N) and (l_prime == l)):
                                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(-k_1_prime)%self._N,:,m_prime]) *\
                                                        reduced_eigenvectors[spins[0],(-k_1)%self._N,:,m] *\
                                                        np.conj(reduced_eigenvectors[spins[1],(-k_2_prime)%self._N,:,n_prime]) *\
                                                        reduced_eigenvectors[spins[1],(-k_2)%self._N,:,n]) * s[0] * s[1]
                                                        
                                                self._R[:,R_row_index,R_column_index] += R_sum / self._N
                                                
                                                R_column_index += 1
                                                
                            R_row_index += 1
                            
        self._R[:,np.arange(R_dimensionality),np.arange(R_dimensionality)] += 1.5 * self._epsilon
        
        # Apply the second correction term from the matrix elements if the
        # spins of the two electron operators are the same.
        # Assume one flat band for now.
        if (spins[0] == spins[1]):
            q_1 = np.broadcast_to(np.arange(self._N), (self._N, self._N))
            q_2 = q_1.flatten()
            q_1 = q_1.T.flatten()
            rearranged_rows = q_2 * self._N + q_1
            self._R -= self._R[:,rearranged_rows]
            
        # Convert 3-dimensional array, self._R, to a list of matrices at the different p
        self._R = list(self._R)
        
        # Remove duplicate and non-independent states, assuming one flat band.
        # Only needs to be considered when the electron operator spins are the same.
        for p_value in range(self._N):
            independent_states = []
            allowed_indices = []
            # Disallowed states occur only when the spins are the same
            if (spins[0] == spins[1]):
                for k_1 in range(self._N):
                    for k_2 in range(self._N):
                        # Sort to remove operator order dependence
                        state = sorted([(-k_1) % self._N, (-k_2) % self._N])
                        if state in independent_states:
                            # This state is not independent (it has already been counted)
                            independent = False
                        else:
                            independent_states.append(state)
                            independent = True
                            
                        # Now check if the state is forbidden by the PEP (momenta are the same)
                        if ((-k_1) % self._N == (-k_2) % self._N):
                            allowed = False
                        else:
                            allowed = True
                            
                        # If the state is independent and allowed, it is kept in the matrix
                        if ((independent == True) and (allowed == True)):
                            allowed_indices.append(True)
                        else:
                            allowed_indices.append(False)
                        
            else:
                # All states are allowed if the electron operator spins are different
                allowed_indices = np.array([True] * self._N**2)
                
            # Project down the matrices
            self._R[p_value] = self._R[p_value][allowed_indices][:,allowed_indices]
        
        self._trion_mixed_energies = []
        
        # Calculate the energy spectrum
        for p_value in range(self._N):
            energies = LA.eigvalsh(self._R[p_value])
            self._trion_mixed_energies.append(self._mod_U * energies)
            
        if (plot == True):
            scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
            scaled_k = np.roll(scaled_k, -(self._N // 2))
            for i, k_value in enumerate(scaled_k):
                length = len(self._trion_mixed_energies[i])
                plt.plot(np.ones(length)*k_value, self._trion_mixed_energies[i], ".", color="red")
            
            plt.plot(np.array([-np.pi, np.pi]), 1.5*self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$\frac{3}{2} \epsilon |U|$")
                
            plt.grid(True)
            plt.legend()
            plt.title("Trion (2 Electrons; 1 Hole) Excitation Spectrum", fontsize=18)
            plt.xlabel(r"$pa$")
            plt.ylabel("Energy")
            plt.show()
            
    def four_electrons_sparse(self, flat_bands, spins, num_eigenvalues=5, plot=True):
        # Check the bands all lie at the same energy. Terminate otherwise.
        self.check_band_degeneracy(flat_bands)
        # Epsilon from the UPC can now be assigned
        self._N_f = len(flat_bands)
        self._epsilon = self._N_f / self._N_orb
        
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,:,flat_bands]
        # Check the model obeys the UPC. Terminate otherwise.
        self.check_UPC(reduced_eigenvectors)
        
        # Define the three spins (0 maps to 1, 1 maps to -1 for up and down respectively)
        # sigma = index 0; sigma prime = index 1; sigma double prime = index 2
        spins = np.array(spins)
        s = -2*spins + 1
        
        R_dimensionality = self._N**3
        
        # Calculate the matrices one at a time to save space; don't store all
        # of them like before (i.e. don't evaluate at all p at once)
        # This will increase processing time a bit but will allow the use of
        # sparse matrices
        self._four_electrons_energies = np.zeros((self._N, num_eigenvalues))
        
        q = np.arange(self._N)
        q=np.broadcast_to(q,(self._N,self._N,self._N))
        q_1 = np.array([[i]*self._N**2 for i in range(self._N)]).flatten()
        q_2 = np.einsum("ijk->ikj",q).flatten()
        q_3 = q.flatten()
        
        # Assume one flat band for now
        for p in range(self._N):
            row_indices = []
            col_indices = []
            values = []
            R_row_index = 0
            for k_1_prime in range(self._N):
                for k_2_prime in range(self._N):
                    for k_3_prime in range(self._N):
                        R_column_index = 0
                        for k_1 in range(self._N):
                            for k_2 in range(self._N):
                                for k_3 in range(self._N):
                                    R_sum = 0.j
                                    
                                    if ((k_2_prime == k_2) and (k_3_prime == k_3)):
                                        R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_prime+k_2_prime+k_3_prime)%self._N,:,0]) *\
                                            reduced_eigenvectors[spins[0],(p+k_1+k_2+k_3)%self._N,:,0] *\
                                            np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,0]) *\
                                            reduced_eigenvectors[spins[1],(-k_1)%self._N,:,0]) * s[0] * s[1]
                                            
                                    if ((k_1_prime == k_1) and (k_3_prime == k_3)):
                                        R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_prime+k_2_prime+k_3_prime)%self._N,:,0]) *\
                                            reduced_eigenvectors[spins[0],(p+k_1+k_2+k_3)%self._N,:,0] *\
                                            np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,0]) *\
                                            reduced_eigenvectors[spins[2],(-k_2)%self._N,:,0]) * s[0] * s[2]
                                            
                                    if ((k_1_prime == k_1) and (k_2_prime == k_2)):
                                        R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_prime+k_2_prime+k_3_prime)%self._N,:,0]) *\
                                            reduced_eigenvectors[spins[0],(p+k_1+k_2+k_3)%self._N,:,0] *\
                                            np.conj(reduced_eigenvectors[spins[3],(-k_3_prime)%self._N,:,0]) *\
                                            reduced_eigenvectors[spins[3],(-k_3)%self._N,:,0]) * s[0] * s[3]
                                            
                                    if ((k_1_prime+k_2_prime+k_3_prime)%self._N == (k_1+k_2+k_3)%self._N):
                                        if (k_3_prime == k_3):
                                            R_sum += np.sum(np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,0]) *\
                                                reduced_eigenvectors[spins[1],(-k_1)%self._N,:,0] *\
                                                np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,0]) *\
                                                reduced_eigenvectors[spins[2],(-k_2)%self._N,:,0]) * s[1] * s[2]
                                                
                                        if (k_2_prime == k_2):
                                            R_sum += np.sum(np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,0]) *\
                                                reduced_eigenvectors[spins[1],(-k_1)%self._N,:,0] *\
                                                np.conj(reduced_eigenvectors[spins[3],(-k_3_prime)%self._N,:,0]) *\
                                                reduced_eigenvectors[spins[3],(-k_3)%self._N,:,0]) * s[1] * s[3]
                                                
                                        if (k_1_prime == k_1):
                                            R_sum += np.sum(np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,0]) *\
                                                reduced_eigenvectors[spins[2],(-k_2)%self._N,:,0] *\
                                                np.conj(reduced_eigenvectors[spins[3],(-k_3_prime)%self._N,:,0]) *\
                                                reduced_eigenvectors[spins[3],(-k_3)%self._N,:,0]) * s[2] * s[3]
                                                
                                    if (R_sum != 0.j):
                                        # Specify only if it is non-zero, otherwise
                                        # store as an implicit zero.
                                        row_indices.append(R_row_index)
                                        col_indices.append(R_column_index)
                                        values.append(R_sum / self._N)
                                        
                                    if (R_row_index == R_column_index):
                                        # Add the constant shift on the diagonal
                                        row_indices.append(R_row_index)
                                        col_indices.append(R_column_index)
                                        values.append(2. * self._epsilon)
                                                
                                    R_column_index += 1
                        
                        R_row_index += 1
                        
            # Construct R as a sparse CSR matrix
            R = sparse.csr_array((values, (row_indices, col_indices)), shape=(R_dimensionality, R_dimensionality))
            
            # Apply corrections for the PEP
            q_rearranged = (-p-q_1-q_2-q_3) % self._N
            # * 1. to create a copy, not a reference to the original matrix
            R_copy = R.copy()
            
            # There is no meaning to this if statement; it just allows the huge
            # block of if statements to be collapsed
            if (True==True):
                # 1
                if ((spins[0] == spins[3]) and (spins[1] == spins[2])):
                    rearranged_rows = q_2*self._N**2 + q_1*self._N + q_rearranged
                    R += R_copy[rearranged_rows,:]
                
                # 2
                if (spins[0] == spins[1] == spins[2] == spins[3]):
                    rearranged_rows = q_3*self._N**2 + q_1*self._N + q_rearranged
                    R -= R_copy[rearranged_rows,:]
                
                # 3
                if (spins[0] == spins[3]): 
                    rearranged_rows = q_1*self._N**2 + q_2*self._N + q_rearranged
                    R -= R_copy[rearranged_rows,:]
                    
                # 4
                if ((spins[0] == spins[3]) and (spins[0] == spins[2]) and (spins[2] == spins[3])):
                    rearranged_rows = q_1*self._N**2 + q_3*self._N + q_rearranged
                    R += R_copy[rearranged_rows,:]
                    
                # 5
                if ((spins[0] == spins[3]) and (spins[0] == spins[1]) and (spins[1] == spins[3])):
                    rearranged_rows = q_3*self._N**2 + q_2*self._N + q_rearranged
                    R += R_copy[rearranged_rows,:]
                    
                # 6
                if (spins[0] == spins[1] == spins[2] == spins[3]):
                    rearranged_rows = q_2*self._N**2 + q_3*self._N + q_rearranged
                    R -= R_copy[rearranged_rows,:]
                    
                # 7
                if (spins[0] == spins[1] == spins[2] == spins[3]):
                    rearranged_rows = q_2*self._N**2 + q_rearranged*self._N + q_1
                    R -= R_copy[rearranged_rows,:]
                
                # 8
                if ((spins[0] == spins[2]) and (spins[1] == spins[3])):
                    rearranged_rows = q_3*self._N**2 + q_rearranged*self._N + q_1
                    R += R_copy[rearranged_rows,:]
                    
                # 9
                if ((spins[0] == spins[2]) and (spins[2] == spins[3]) and (spins[0] == spins[3])):
                    rearranged_rows = q_1*self._N**2 + q_rearranged*self._N + q_2
                    R += R_copy[rearranged_rows,:]
                
                # 10
                if (spins[0] == spins[2]):
                    rearranged_rows = q_1*self._N**2 + q_rearranged*self._N + q_3
                    R -= R_copy[rearranged_rows,:]
                    
                # 11:
                if (spins[0] == spins[1] == spins[2] == spins[3]):
                    rearranged_rows = q_3*self._N**2 + q_rearranged*self._N + q_2
                    R -= R_copy[rearranged_rows,:]
                
                # 12
                if ((spins[0] == spins[2]) and (spins[0] == spins[1]) and (spins[1] == spins[2])):
                    rearranged_rows = q_2*self._N**2 + q_rearranged*self._N + q_3
                    R += R_copy[rearranged_rows,:]
                
                # 13
                if ((spins[0] == spins[1]) and (spins[1] == spins[3]) and (spins[0] == spins[3])):
                    rearranged_rows = q_rearranged*self._N**2 + q_2*self._N + q_1
                    R += R_copy[rearranged_rows,:]
                    
                # 14
                if (spins[0] == spins[1] == spins[2] == spins[3]):
                    rearranged_rows = q_rearranged*self._N**2 + q_3*self._N + q_1
                    R -= R_copy[rearranged_rows,:]
                    
                # 15
                if (spins[0] == spins[1] == spins[2] == spins[3]):
                    rearranged_rows = q_rearranged*self._N**2 + q_1*self._N + q_2
                    R -= R_copy[rearranged_rows,:]
                    
                # 16
                if ((spins[0] == spins[1]) and (spins[1] == spins[2]) and (spins[0] == spins[2])):
                    rearranged_rows = q_rearranged*self._N**2 + q_1*self._N + q_3
                    R += R_copy[rearranged_rows,:]
                    
                # 17
                if ((spins[0] == spins[1]) and (spins[2] == spins[3])):
                    rearranged_rows = q_rearranged*self._N**2 + q_3*self._N + q_2
                    R += R_copy[rearranged_rows,:]
                    
                # 18
                if (spins[0] == spins[1]):
                    rearranged_rows = q_rearranged*self._N**2 + q_2*self._N + q_3
                    R -= R_copy[rearranged_rows,:]
                    
                # 19
                if (spins[1] == spins[3]):
                    rearranged_rows = q_3*self._N**2 + q_2*self._N + q_1
                    R -= R_copy[rearranged_rows,:]
                    
                # 20
                if ((spins[1] == spins[3]) and (spins[1] == spins[2]) and (spins[2] == spins[3])):
                    rearranged_rows = q_2*self._N**2 + q_3*self._N + q_1
                    R += R_copy[rearranged_rows,:]
                    
                # 21
                if ((spins[1] == spins[2]) and (spins[2] == spins[3]) and (spins[1] == spins[3])):
                    rearranged_rows = q_3*self._N**2 + q_1*self._N + q_2
                    R += R_copy[rearranged_rows,:]
                    
                # 22
                if (spins[1] == spins[2]):
                    rearranged_rows = q_2*self._N**2 + q_1*self._N + q_3
                    R -= R_copy[rearranged_rows,:]
                    
                # 23
                if (spins[2] == spins[3]):
                    rearranged_rows = q_1*self._N**2 + q_3*self._N + q_2
                    R -= R_copy[rearranged_rows,:]
                        
            # Remove duplicate (non-independent) states and those forbidden by the PEP
            # Assume only one flat band for now
            independent_states = []
            allowed_indices = []
            index = 0
            for k_1 in range(self._N):
                for k_2 in range(self._N):
                    for k_3 in range(self._N):
                        # Sorting removes (some) dependence on operator order
                        momentum_state = np.array([(p + k_1 + k_2 + k_3)%self._N, (-k_1)%self._N, (-k_2)%self._N, (-k_3)%self._N])
                        sorted_indices = momentum_state.argsort()
                        momentum_state = momentum_state[sorted_indices]
                        spin_state = spins[sorted_indices]
                            
                        unique_momenta = np.unique(momentum_state)
                        # Everything is automatically sorted if all momenta are different;
                        # check otherwise that spins are also sorted
                        if (len(unique_momenta) != 4):
                            current = 0
                            for momentum in unique_momenta:
                                count = sum(momentum_state == momentum)
                                spin_state[current:current+count] = np.sort(spin_state[current:current+count])
                                current += count
                        
                        state = list(momentum_state) + list(spin_state)
                        if state in independent_states:
                            # This state is not independent (it has already been counted)
                            independent = False
                        else:
                            independent_states.append(state)
                            independent = True
                            
                        # Check if the state is forbidden by the PEP (no need to check
                        # if the state is already non-independent)
                        if (independent == True):
                            # This is probably not the fastest way to check;
                            # come back to this later
                            if (np.allclose(0., R[[index],:].toarray())):
                                # Disallowed by the PEP if True
                                allowed = False
                            else:
                                allowed = True
                        else:
                            allowed = False
                            
                        if ((independent == True) and (allowed == True)):
                            allowed_indices.append(True)
                        else:
                            allowed_indices.append(False)
                            
                        index += 1
                        
            # Project down R
            R = R[allowed_indices,:][:,allowed_indices]
            
            # Find only some of the lowest energies (eigenvalues)
            energies = sparse.linalg.eigsh(R, k=num_eigenvalues, which="SA", return_eigenvectors=False)
            self._four_electrons_energies[p] += energies
            
            print("p = %.i complete" % (p))
            
        if (plot==True):
            scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
            scaled_k = np.roll(scaled_k, -(self._N // 2))
            for i, k_value in enumerate(scaled_k):
                length = len(self._four_electrons_energies[i])
                plt.plot(np.ones(length)*k_value, self._four_electrons_energies[i], ".", color="green")
            
            plt.plot(np.array([-np.pi, np.pi]), 2.*self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$2 \epsilon |U|$")
                
            plt.grid(True)
            plt.legend()
            plt.title("4 Electrons Excitation Spectrum", fontsize=18)
            plt.xlabel(r"$pa$")
            plt.ylabel("Energy")
            plt.show()
    
    def plot_band_structure(self):
        """
        Plots the reduced-zone band structure resulting from the specified
        Hamiltonian. Exists for diagnostic purposes.

        Returns
        -------
        None.

        """
        
        scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
        
        fig, (ax1, ax2) = plt.subplots(2)
        fig.set_figwidth(7)
        fig.set_figheight(8)
        ax1.tick_params(axis="x", labelsize=14)
        ax1.tick_params(axis="y", labelsize=14)
        ax2.tick_params(axis="x", labelsize=14)
        ax2.tick_params(axis="y", labelsize=14)
        
        ax1.set_title(r"Band Structure ($\uparrow$)", fontsize=16)
        ax2.set_title(r"Band Structure ($\downarrow$)", fontsize=16)
        ax1.set_xlabel(r"$ka$", fontsize=14)
        ax2.set_xlabel(r"$ka$", fontsize=14)
        ax1.set_ylabel("Energy", fontsize=14)
        ax2.set_ylabel("Energy", fontsize=14)
        
        for i in range(self._N_orb):
            ax1.plot(scaled_k, np.roll(self._eigenvalues[0,:,i], self._N // 2), ".", label="Band %.i" % (i))
            ax2.plot(scaled_k, np.roll(self._eigenvalues[1,:,i], self._N // 2), ".", label="Band %.i" % (i))
            
        ax1.legend(fontsize=14)
        ax2.legend(fontsize=14)
        ax1.grid(True)
        ax2.grid(True)
        
        fig.show()