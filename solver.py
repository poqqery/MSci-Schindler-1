# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 01:31:09 2023

@author: Madhuwrit, Louis
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

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
            this number small (<= 20) if evaluating trions. N must be even!
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
        self._k_samples = np.linspace(0., 2.*np.pi, self._N + 1)[:-1]
        self._k_samples /= self._a
        
        # Find the flat bands, if any exist
        self._flat_bands_up, self._flat_bands_down = self.identify_flat_bands()
        
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
        # 2 spins, N wavevectors, num_bands eigenvalues
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
                self._eigenvalues[spin,i] += sample_eigenvalues
                self._eigenvectors[spin,i] += sample_eigenvectors
                
        # Now identify flat bands (if any exist).
        # This algorithm is technically flawed because it depends on
        # np.linalg.eigh returning the eigenvalues of the bands in the
        # same "order" consistently, but it'll probably work so long as bands
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
    
    def charge_1_spectra(self, flat_bands, spin):
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
        
        scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
        # Plot the excitation spectra (should always be flat)
        for i in range(self._charge_1_energies.shape[1]):
            plt.plot(scaled_k, np.roll(self._charge_1_energies[:,i], self._N // 2), ".")
            
        plt.grid(True)
        plt.title("Excitation Spectra", fontsize=18)
        plt.xlabel(r"$ka$")
        plt.ylabel("Energy")
        plt.show()
        
    def charge_2_spectra(self, flat_bands):
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
        
        scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
        # Plot the excitation spectra
        for i in range(self._charge_2_energies.shape[1]):
            plt.plot(scaled_k, np.roll(self._charge_2_energies[:,i], self._N // 2), ".")
            
        plt.plot(np.array([-np.pi, np.pi]), self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$\epsilon |U|$")
            
        plt.grid(True)
        plt.legend()
        plt.title("Excitation Spectra", fontsize=18)
        plt.xlabel(r"$pa$")
        plt.ylabel("Energy")
        plt.show()
        
    def trion_3_electrons_spectra(self, flat_bands, spins):
        
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
        
        trion_3_electron_energies = []
        
        # Number of rows and columns in R
        R_dimensionality = self._N**2 * self._N_f**3
        
        # Iterate over each p
        for p in range(self._N):
            R = (1.5 + 0.j)*self._epsilon*self._mod_U * np.identity(R_dimensionality)
            
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
                                                    if ((k_1_prime+k_2_prime == k_1+k_2) and (m_prime == m)):
                                                        R_sum += sum(np.conj(reduced_eigenvectors[s[1],(-k_1_prime)%self._N,:,n_prime]) *\
                                                            reduced_eigenvectors[s[1],(-k_1)%self._N,:,n] *\
                                                            np.conj(reduced_eigenvectors[s[2],(-k_2_prime)%self._N,:,l_prime]) *\
                                                            reduced_eigenvectors[s[2],(-k_2)%self._N,:,l]) * s[1] * s[2]
                                                    
                                                    if ((k_1_prime == k_1) and (n_prime == n)):
                                                        R_sum += sum(np.conj(reduced_eigenvectors[s[0],(p+k_1+k_2_prime)%self._N,:,m_prime]) *\
                                                            reduced_eigenvectors[s[0],(p+k_1+k_2)%self._N,:,m] *\
                                                            np.conj(reduced_eigenvectors[s[2],(-k_2_prime)%self._N,:,l_prime]) *\
                                                            reduced_eigenvectors[s[2],(-k_2)%self._N,:,l]) * s[0] * s[2]
                                                            
                                                    if ((k_2_prime == k_2) and (l_prime == l)):
                                                        R_sum += sum(np.conj(reduced_eigenvectors[s[0],(p+k_1_prime+k_2)%self._N,:,m_prime]) *\
                                                            reduced_eigenvectors[s[0],(p+k_1+k_2)%self._N,:,m] *\
                                                            np.conj(reduced_eigenvectors[s[1],(-k_1_prime)%self._N,:,n_prime]) *\
                                                            reduced_eigenvectors[s[1],(-k_1)%self._N,:,n]) * s[0] * s[1]
                                                            
                                                    R[R_row_index,R_column_index] += (self._mod_U / self._N) * R_sum
                                                    
                                                    R_column_index += 1
                                
                                R_row_index += 1
                                
            energies = LA.eigvalsh(R)
            trion_3_electron_energies.append(energies)
            
        self._trion_3_electron_energies = np.array(trion_3_electron_energies)
            
        scaled_k = np.linspace(-np.pi, np.pi, self._N + 1)[:-1]
        # Plot the excitation spectra
        for i in range(self._trion_3_electron_energies.shape[1]):
            plt.plot(scaled_k, np.roll(self._trion_3_electron_energies[:,i], self._N // 2), ".", color="red")
            
        plt.plot(np.array([-np.pi, np.pi]), 1.5*self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$\frac{3}{2}\epsilon |U|$")
            
        plt.grid(True)
        plt.legend()
        plt.title("Trion (3 Electrons) Excitation Spectra", fontsize=18)
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