# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 01:31:09 2023

@author: Madhuwrit, Louis
"""

import numpy as np
import numpy.linalg as LA
from scipy import sparse
import matplotlib.pyplot as plt
from copy import deepcopy

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
        NoFlatBandException
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
                sample_eigenvectors = 0.5*np.array([[1.-np.exp(-1.j * self._k_samples[i]), 0.], [1.+np.exp(-1.j * self._k_samples[i]), 0.]])
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
    
    def find_valid_sum_terms(self):
        
        # 8-bit to save memory (shouldn't really need any bigger, anyway)
        p = np.arange(self._N, dtype="int8")
        k_prime_outer = np.add.outer(p, p) % self._N
        # k_prime_outer varies over the first 2 indices
        k_prime_outer = np.broadcast_to(k_prime_outer, (self._N, self._N, self._N, self._N))
        # k_outer varies over the last 2 indices
        k_outer = np.einsum("ijkl->klij", k_prime_outer)
        k_delta_tensor = k_prime_outer == k_outer
        # This array gives the values that need to be summed over in the sum
        # with a delta spanning four k indices (the complicated one)
        self._true_indices = np.array(np.nonzero(k_delta_tensor)).T
    
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
            scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
            # Plot the excitation spectra
            for i in range(self._charge_2_energies.shape[1]):
                plt.plot(scaled_k, np.roll(self._charge_2_energies[:,i], self._N // 2), ".", color="blue")
                
            plt.plot(np.array([-np.pi, np.pi]), self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$\epsilon |U|$")
                
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
            for p_value in range(self._N):
                q = np.arange(self._N)
                rearranged = (-p_value-q) % self._N
                self._R[p_value] -= self._R[p_value,rearranged]
                
        # Convert 3-dimensional array, self._R, to a list of matrices at the different p
        self._R = list(self._R)
                
        # Remove duplicate (non-independent) states and those forbidden by the PEP
        # (these will have entirely zero rows/columns at the index corresponding
        # to that state).
        # Ignore the multiple-flat-band case for now (it considerably complicates
        # the indexing of states).
        for p_value in range(self._N):
            independent_states = []
            allowed_indices = []
            # Disallowed states occur only when the spins are the same
            if (spins[0] == spins[1]):
                for k_value in range(self._N):
                    # Go through each state at this p
                    # Sorting removes dependence on the operator order of gamma dagger
                    state = sorted([(p_value + k_value) % self._N, (-k_value) % self._N])
                    if state in independent_states:
                        # This state is not independent (it has already been counted)
                        independent = False
                    else:
                        independent_states.append(state)
                        independent = True
    
                    # Now check if the state is forbidden by the PEP (momenta are the same)
                    if ((p_value + k_value) % self._N == (-k_value) % self._N):
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
                #allowed_indices = [True for i in range(self._N)]
                allowed_indices = np.array([True] * self._N)
                    
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
        
    def trions_3_electrons(self, flat_bands, spins, plot=True):
        
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
                                                R_sum = 0.j
                                                if (((k_1_prime+k_2_prime)%self._N == (k_1+k_2)%self._N) and (m_prime == m)):
                                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,n_prime]) *\
                                                        reduced_eigenvectors[spins[1],(-k_1)%self._N,:,n] *\
                                                        np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,l_prime]) *\
                                                        reduced_eigenvectors[spins[2],(-k_2)%self._N,:,l]) * s[1] * s[2]
                                                
                                                if ((k_1_prime == k_1) and (n_prime == n)):
                                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1+k_2_prime)%self._N,:,m_prime]) *\
                                                        reduced_eigenvectors[spins[0],(p+k_1+k_2)%self._N,:,m] *\
                                                        np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,l_prime]) *\
                                                        reduced_eigenvectors[spins[2],(-k_2)%self._N,:,l], axis=1) * s[0] * s[2]
                                                        
                                                if ((k_2_prime == k_2) and (l_prime == l)):
                                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_prime+k_2)%self._N,:,m_prime]) *\
                                                        reduced_eigenvectors[spins[0],(p+k_1+k_2)%self._N,:,m] *\
                                                        np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,n_prime]) *\
                                                        reduced_eigenvectors[spins[1],(-k_1)%self._N,:,n], axis=1) * s[0] * s[1]
                                                        
                                                self._R[:,R_row_index,R_column_index] += R_sum / self._N
                                                
                                                R_column_index += 1
                            
                            R_row_index += 1
                            
        self._R[:,np.arange(R_dimensionality),np.arange(R_dimensionality)] += 1.5 * self._epsilon
                            
        # Apply the multitude of corrections to R (assume only 1 flat band for now)
        q = np.arange(self._N)
        q_1 = np.broadcast_to(q, (self._N, self._N))
        q_2 = q_1.flatten()
        q_1 = q_1.T.flatten()
        for p_value in range(self._N):
            q_rearranged = (-p_value-q_1-q_2) % self._N
            # * 1. to create a copy, not a reference to the original matrix
            R_copy = self._R[p_value] * 1.
            if (spins[0] == spins[1]):
                rearranged_rows = q_rearranged * self._N + q_2
                self._R[p_value] -= R_copy[rearranged_rows]
                
            if (spins[0] == spins[2]):
                rearranged_rows = q_1 * self._N + q_rearranged
                self._R[p_value] -= R_copy[rearranged_rows]
                
            if (spins[1] == spins[2]):
                rearranged_rows = q_2 * self._N + q_1
                self._R[p_value] -= R_copy[rearranged_rows]
                
            if (spins[0] == spins[1] == spins[2]):
                rearranged_rows = q_rearranged * self._N + q_1
                self._R[p_value] += R_copy[rearranged_rows]
                rearranged_rows = q_2 * self._N + q_rearranged
                self._R[p_value] += R_copy[rearranged_rows]
                
        # Convert 3-dimensional array, self._R, to a list of matrices at the different p
        self._R = list(self._R)
        
        # Remove duplicate (non-independent) states and those forbidden by the PEP
        # Assume only one flat band for now
        for p_value in range(self._N):
            independent_states = []
            allowed_indices = []
            index = 0
            for k_1 in range(self._N):
                for k_2 in range(self._N):
                    # Sorting removes (some) dependence on operator order
                    momentum_state = np.array([(p_value + k_1 + k_2)%self._N, (-k_1)%self._N, (-k_2)%self._N])
                    sorted_indices = momentum_state.argsort()
                    momentum_state = momentum_state[sorted_indices]
                    spin_state = spins[sorted_indices]
                        
                    unique_momenta = np.unique(momentum_state)
                    # Everything is automatically sorted if all three momenta are different;
                    # check otherwise that spins are also sorted
                    if (len(unique_momenta) != 3):
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
                        if (np.allclose(0., self._R[p_value][index])):
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
                        
            # Project down the matrix to remove forbidden/non-independent states
            self._R[p_value] = self._R[p_value][allowed_indices][:,allowed_indices]
            
        self._trion_electrons_energies = []
        
        # Calculate the energy spectrum (finally)
        for p_value in range(self._N):
            energies = LA.eigvalsh(self._R[p_value])
            self._trion_electrons_energies.append(self._mod_U * energies)
            
        if (plot == True):
            scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
            scaled_k = np.roll(scaled_k, -(self._N // 2))
            for i, k_value in enumerate(scaled_k):
                length = len(self._trion_electrons_energies[i])
                plt.plot(np.ones(length)*k_value, self._trion_electrons_energies[i], ".", color="red")
            
            plt.plot(np.array([-np.pi, np.pi]), 1.5*self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$\frac{3}{2} \epsilon |U|$")
                
            plt.grid(True)
            plt.legend()
            plt.title("Trion (3 Electrons) Excitation Spectrum", fontsize=18)
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
        
    def trion_3_electrons_sparse(self, flat_bands, spins, plot=True):
        
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
        s = -2*np.array(spins) + 1
        
        # Number of rows and columns in R
        R_dimensionality =  self._N**2 * self._N_f**3
        
        # First check that the list of non-zero summation terms for the
        # complicated Kronecker delta has been calculated; generate them if
        # they haven't
        if (type(self._true_indices) != np.ndarray):
            self.find_valid_sum_terms()
            
        p = np.arange(self._N)
        # This p_cube thing looks complicated but is needed to quickly evaluate
        # the sums (see below)
        p_cube = np.broadcast_to(p, (self._N, self._N, self._N))
        # R will be populated with lists of sparse matrices (one for each p)
        R = []
        
        for p_val in p:
            # These 3 lists will build the sparse matrix, R
            rows = []
            cols = []
            vals = []
            row_index_count = 0
            for m_prime in range(self._N_f):
                for n_prime in range(self._N_f):
                    for l_prime in range(self._N_f):
                        col_index_count = 0
                        for m in range(self._N_f):
                            for n in range(self._N_f):
                                for l in range(self._N_f):
                                    # Evaluate the first sum with the more complex
                                    # requirements from the Kronecker delta
                                    if (m_prime == m):
                                        k_1_prime = self._true_indices[:,0]
                                        k_2_prime = self._true_indices[:,1]
                                        k_1 = self._true_indices[:,2]
                                        k_2 = self._true_indices[:,3]
                                        rows.extend(row_index_count*self._N**2 + k_1_prime*self._N + k_2_prime)
                                        cols.extend(col_index_count*self._N**2 + k_1*self._N + k_2)
                                        vals.extend((np.sum(np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,n_prime]) *\
                                            reduced_eigenvectors[spins[1],(-k_1)%self._N,:,n] *\
                                            np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,l_prime]) *\
                                            reduced_eigenvectors[spins[2],(-k_2)%self._N,:,l], axis=1) / self._N) * s[1] * s[2])
                                            
                                    # Evaluate the second sum (k_1_prime = k_1)
                                    if (n_prime == n):
                                        # What's going on here is pretty confusing;
                                        # I can explain in person if you want
                                        k_1 = np.einsum("ijk->kij", p_cube).flatten()
                                        k_2_prime = np.einsum("ijk->ikj", p_cube).flatten()
                                        k_2 = p_cube.flatten()
                                        rows.extend(row_index_count*self._N**2 + k_1*self._N + k_2_prime)
                                        cols.extend(col_index_count*self._N**2 + k_1*self._N + k_2)
                                        vals.extend((np.sum(np.conj(reduced_eigenvectors[spins[0],(p_val+k_1+k_2_prime)%self._N,:,m_prime]) *\
                                            reduced_eigenvectors[spins[0],(p_val+k_1+k_2)%self._N,:,m] *\
                                            np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,l_prime]) *\
                                            reduced_eigenvectors[spins[2],(-k_2)%self._N,:,l], axis=1) / self._N) * s[0] *s[2])
                                            
                                    # Evaluate the third sum (k_2_prime = k_2)
                                    if (l_prime == l):
                                        k_2 = np.einsum("ijk->kij", p_cube).flatten()
                                        k_1_prime = np.einsum("ijk->ikj", p_cube).flatten()
                                        k_1 = p_cube.flatten()
                                        rows.extend(row_index_count*self._N**2 + k_1_prime*self._N + k_2)
                                        cols.extend(col_index_count*self._N**2 + k_1*self._N + k_2)
                                        vals.extend((np.sum(np.conj(reduced_eigenvectors[spins[0],(p_val+k_1_prime+k_2)%self._N,:,m_prime]) *\
                                            reduced_eigenvectors[spins[0],(p_val+k_1+k_2)%self._N,:,m] *\
                                            np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,n_prime]) *\
                                            reduced_eigenvectors[spins[1],(-k_1)%self._N,:,n], axis=1) / self._N) * s[0] * s[1])
                                        
                                    col_index_count += 1
                                    
                        row_index_count += 1
                        
            # Now build the sparse matrix
            R.append(sparse.csr_array((vals, (rows, cols)), shape=(R_dimensionality, R_dimensionality)))
        
        # Find the 5 lowest eigenvalues (energies) at each p
        self._trion_electrons_energies = np.zeros((self._N, 5))
        for i in range(self._N):
            # Don't need the eigenvectors, but the eigen solver returns them
            # anyway
            energies, vectors = sparse.linalg.eigsh(R[i], k=5, which="SA")
            self._trion_electrons_energies[i] += self._mod_U * (1.5*self._epsilon + energies)
            
        if (plot == True):
            scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
            # Plot the lowest excitation spectra
            for i in range(5):
                plt.plot(scaled_k, np.roll(self._trion_electrons_energies[:,i], self._N // 2), ".", color="blue")
                
            plt.plot(np.array([-np.pi, np.pi]), 1.5*self._mod_U*self._epsilon*np.ones(2),\
                "--", color="black", label=r"$\frac{3}{2}\epsilon |U|$")
                
            plt.grid(True)
            plt.legend()
            plt.title("Trion (3 Electrons) Excitation Spectrum", fontsize=18)
            plt.xlabel(r"$pa$")
            plt.ylabel("Energy")
            plt.show()
    
    def charge_4_electrons(self, flat_bands, spins, plot=True):
            
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
        R_dimensionality = self._N**3
        self._R = np.zeros((self._N, R_dimensionality, R_dimensionality), dtype="complex128")
        
        # Calculate a component of R for all p at once
        p = np.arange(self._N)
        
        # (I honestly can't see any way right now to avoid 10 for loops...)
        # Counts the current row of R being filled in
        R_row_index = 0
        n=0 #all the bands will be index 0
        
        for k_1_prime in range(self._N):
            for k_2_prime in range(self._N):
                for k_3_prime in range(self._N):
                    # Counts the current column of R being filled in
                    R_column_index = 0
                    for k_1 in range(self._N):
                        for k_2 in range(self._N):
                            for k_3 in range(self._N):
                                R_sum = 0.j
                                
                                if (k_2_prime == k_2) and (k_3_prime == k_3):
                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_prime+k_2_prime+k_3_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[0],(p+k_1+k_2+k_3)%self._N,:,n] *\
                                        np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[1],(-k_1)%self._N,:,n], axis=1) * s[0] * s[1]
                                        
                                if (k_1_prime == k_1) and (k_3_prime == k_3):
                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_prime+k_2_prime+k_3_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[0],(p+k_1+k_2+k_3)%self._N,:,n] *\
                                        np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[2],(-k_2)%self._N,:,n], axis=1) * s[0] * s[2]
                                
                                if (k_1_prime == k_1) and (k_2_prime == k_2):
                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[0],(p+k_1_prime+k_2_prime+k_3_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[0],(p+k_1+k_2+k_3)%self._N,:,n] *\
                                        np.conj(reduced_eigenvectors[spins[3],(-k_3_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[3],(-k_3)%self._N,:,n], axis=1) * s[0] * s[3]
                                
                                if ((k_1_prime+k_2_prime+k_3_prime)%self._N == (k_1+k_2+k_3)%self._N) and (k_3_prime == k_3):
                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[1],(-k_1)%self._N,:,n] *\
                                        np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[2],(-k_2)%self._N,:,n]) * s[1] * s[2]
                                        
                                if ((k_1_prime+k_2_prime+k_3_prime)%self._N == (k_1+k_2+k_3)%self._N) and (k_2_prime == k_2):
                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[1],(-k_1_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[1],(-k_1)%self._N,:,n] *\
                                        np.conj(reduced_eigenvectors[spins[3],(-k_3_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[3],(-k_3)%self._N,:,n]) * s[1] * s[3]
                                        
                                if ((k_1_prime+k_2_prime+k_3_prime)%self._N == (k_1+k_2+k_3)%self._N) and (k_1_prime == k_1):
                                    R_sum += np.sum(np.conj(reduced_eigenvectors[spins[2],(-k_2_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[2],(-k_2)%self._N,:,n] *\
                                        np.conj(reduced_eigenvectors[spins[3],(-k_3_prime)%self._N,:,n]) *\
                                        reduced_eigenvectors[spins[3],(-k_3)%self._N,:,n]) * s[2] * s[3]
                                        
                                self._R[:,R_row_index,R_column_index] += R_sum / self._N
                                                
                                R_column_index += 1
                            
                    R_row_index += 1
                            
        self._R[:,np.arange(R_dimensionality),np.arange(R_dimensionality)] += 2 * self._epsilon
        
        q = np.arange(self._N)
        q=np.broadcast_to(q,(self._N,self._N,self._N))
        q_1 = np.array([[i]*self._N**2 for i in range(self._N)]).flatten()
        q_2 = np.einsum("ijk->ikj",q).flatten()
        q_3 = q.flatten()
        
        for p_value in range(self._N):
            q_rearranged = (-p_value-q_1-q_2-q_3) % self._N
            # * 1. to create a copy, not a reference to the original matrix
            R_copy = self._R[p_value] * 1.
            
            # 1
            if ((spins[0] == spins[3]) and (spins[1] == spins[2])):
                rearranged_rows = q_2*self._N**2 + q_1*self._N + q_rearranged
                self._R[p_value] += R_copy[rearranged_rows]
            
            # 2
            if (spins[0] == spins[1] == spins[2] == spins[3]):
                rearranged_rows = q_3*self._N**2 + q_1*self._N + q_rearranged
                self._R[p_value] -= R_copy[rearranged_rows]
            
            # 3
            if (spins[0] == spins[3]): 
                rearranged_rows = q_1*self._N**2 + q_2*self._N + q_rearranged
                self._R[p_value] -= R_copy[rearranged_rows]
                
            # 4
            if ((spins[0] == spins[3]) and (spins[0] == spins[2]) and (spins[2] == spins[3])):
                rearranged_rows = q_1*self._N**2 + q_3*self._N + q_rearranged
                self._R[p_value] += R_copy[rearranged_rows]
                
            # 5
            if ((spins[0] == spins[3]) and (spins[0] == spins[1]) and (spins[1] == spins[3])):
                rearranged_rows = q_3*self._N**2 + q_2*self._N + q_rearranged
                self._R[p_value] += R_copy[rearranged_rows]
                
            # 6
            if (spins[0] == spins[1] == spins[2] == spins[3]):
                rearranged_rows = q_2*self._N**2 + q_3*self._N + q_rearranged
                self._R[p_value] -= R_copy[rearranged_rows]
                
            # 7
            if (spins[0] == spins[1] == spins[2] == spins[3]):
                rearranged_rows = q_2*self._N**2 + q_rearranged*self._N + q_1
                self._R[p_value] -= R_copy[rearranged_rows]
            
            # 8
            if ((spins[0] == spins[2]) and (spins[1] == spins[3])):
                rearranged_rows = q_3*self._N**2 + q_rearranged*self._N + q_1
                self._R[p_value] += R_copy[rearranged_rows]
                
            # 9
            if ((spins[0] == spins[2]) and (spins[2] == spins[3]) and (spins[0] == spins[3])):
                rearranged_rows = q_1*self._N**2 + q_rearranged*self._N + q_2
                self._R[p_value] += R_copy[rearranged_rows]
            
            # 10
            if (spins[0] == spins[2]):
                rearranged_rows = q_1*self._N**2 + q_rearranged*self._N + q_3
                self._R[p_value] -= R_copy[rearranged_rows]
                
            # 11:
            if (spins[0] == spins[1] == spins[2] == spins[3]):
                rearranged_rows = q_3*self._N**2 + q_rearranged*self._N + q_2
                self._R[p_value] -= R_copy[rearranged_rows]
            
            # 12
            if ((spins[0] == spins[2]) and (spins[0] == spins[1]) and (spins[1] == spins[2])):
                rearranged_rows = q_2*self._N**2 + q_rearranged*self._N + q_3
                self._R[p_value] += R_copy[rearranged_rows]
            
            # 13
            if ((spins[0] == spins[1]) and (spins[1] == spins[3]) and (spins[0] == spins[3])):
                rearranged_rows = q_rearranged*self._N**2 + q_2*self._N + q_1
                self._R[p_value] += R_copy[rearranged_rows]
                
            # 14
            if (spins[0] == spins[1] == spins[2] == spins[3]):
                rearranged_rows = q_rearranged*self._N**2 + q_3*self._N + q_1
                self._R[p_value] -= R_copy[rearranged_rows]
                
            # 15
            if (spins[0] == spins[1] == spins[2] == spins[3]):
                rearranged_rows = q_rearranged*self._N**2 + q_1*self._N + q_2
                self._R[p_value] -= R_copy[rearranged_rows]
                
            # 16
            if ((spins[0] == spins[1]) and (spins[1] == spins[2]) and (spins[0] == spins[2])):
                rearranged_rows = q_rearranged*self._N**2 + q_1*self._N + q_3
                self._R[p_value] += R_copy[rearranged_rows]
                
            # 17
            if ((spins[0] == spins[1]) and (spins[2] == spins[3])):
                rearranged_rows = q_rearranged*self._N**2 + q_3*self._N + q_2
                self._R[p_value] += R_copy[rearranged_rows]
                
            # 18
            if (spins[0] == spins[1]):
                rearranged_rows = q_rearranged*self._N**2 + q_2*self._N + q_3
                self._R[p_value] -= R_copy[rearranged_rows]
                
            # 19
            if (spins[1] == spins[3]):
                rearranged_rows = q_3*self._N**2 + q_2*self._N + q_1
                self._R[p_value] -= R_copy[rearranged_rows]
                
            # 20
            if ((spins[1] == spins[3]) and (spins[1] == spins[2]) and (spins[2] == spins[3])):
                rearranged_rows = q_2*self._N**2 + q_3*self._N + q_1
                self._R[p_value] += R_copy[rearranged_rows]
                
            # 21
            if ((spins[1] == spins[2]) and (spins[2] == spins[3]) and (spins[1] == spins[3])):
                rearranged_rows = q_3*self._N**2 + q_1*self._N + q_2
                self._R[p_value] += R_copy[rearranged_rows]
                
            # 22
            if (spins[1] == spins[2]):
                rearranged_rows = q_2*self._N**2 + q_1*self._N + q_3
                self._R[p_value] -= R_copy[rearranged_rows]
                
            # 23
            if (spins[2] == spins[3]):
                rearranged_rows = q_1*self._N**2 + q_3*self._N + q_2
                self._R[p_value] -= R_copy[rearranged_rows]
        
        # Convert 3-dimensional array, self._R, to a list of matrices at the different p
        self._R = list(self._R)
        
        # Remove duplicate (non-independent) states and those forbidden by the PEP
        # Assume only one flat band for now
        for p_value in range(self._N):
            independent_states = []
            allowed_indices = []
            index = 0
            for k_1 in range(self._N):
                for k_2 in range(self._N):
                    for k_3 in range(self._N):
                        # Sorting removes (some) dependence on operator order
                        momentum_state = np.array([(p_value + k_1 + k_2 + k_3)%self._N, (-k_1)%self._N, (-k_2)%self._N, (-k_3)%self._N])
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
                            if (np.allclose(0., self._R[p_value][index])):
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
                        
            # Project down the matrix to remove forbidden/non-independent states
            self._R[p_value] = self._R[p_value][allowed_indices][:,allowed_indices]
        
        self._four_electrons_energies = []
        
        # Calculate the energy spectrum (finally)
        for p_value in range(self._N):
            energies = LA.eigvalsh(self._R[p_value])
            self._four_electrons_energies.append(self._mod_U * energies)
            
        if (plot == True):
            scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
            scaled_k = np.roll(scaled_k, -(self._N // 2))
            for i, k_value in enumerate(scaled_k):
                length = len(self._four_electrons_energies[i])
                plt.plot(np.ones(length)*k_value, self._four_electrons_energies[i], ".", color="green")
            
            plt.plot(np.array([-np.pi, np.pi]), 2*self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$2 \epsilon |U|$")
                
            plt.grid(True)
            plt.legend()
            plt.title("4 Electrons Excitation Spectrum", fontsize=18)
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
            R_copy = deepcopy(R)
            
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
            
            plt.plot(np.array([-np.pi, np.pi]), 2*self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$2 \epsilon |U|$")
                
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