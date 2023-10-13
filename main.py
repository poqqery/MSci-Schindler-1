# -*- coding: utf-8 -*-
"""

Run this master file to access simulations of the excitation spectra of the
different cases.

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

#%%

class NoFlatBandException(Exception):
    "Raised in the absence of flat bands."
    pass

class NonDegenerateFlatBandsException(Exception):
    "Raised if chosen flat bands to analyze excitation spectra of are not degenerate."
    pass

class excitation_solver:
    def __init__(self, interaction_strength, hamiltonian, lattice_constant, N=75):
        """
        Solves the low-temperature excitation spectra for the given
        Hamiltonian—if flat bands exist. Currently only works in 1D; need to
        add functionality for 2 or 3 dimensions—if you're mad.

        Parameters
        ----------
        interaction_strength : float
            Characterizes the Hubbard interaction strength.
        hamiltonian : function object
            Function taking a wavevector, k, as input. Generally returns a
            matrix. Assumed to be Hermitian (it damn well better be).
        lattice_constant : float
            Lattice spacing from one unit cell to another.
        N : int
            Number of unit cells. This automatically determines the valid
            wavevectors to sample at: the ones that satisfy periodic boundary
            conditions and maintain independence. Probably don't make this
            massive to prevent long runtimes, especially in more than one
            dimension. The default is 75.

        Returns
        -------
        None.

        """
        
        self._mod_U = interaction_strength
        self._hamiltonian = hamiltonian
        self._a = lattice_constant
        self._N = N
        # Linearly spaced samples of k in the FBZ (currently only 1D)
        # The last k is omitted because it is the same state as the first
        # (Just like a discrete Fourier transform!)
        self._k_samples = np.linspace(-np.pi / self._a, np.pi / self._a, self._N + 1)[:-1]
        
        # Find the flat bands, if any exist
        self._flat_bands = self.identify_flat_bands()
        
    def check_band_degeneracy(self, flat_bands):
        
        # First check that the chosen flat bands to analyze do actually
        # all lie at the same energy
        first_energies = self._eigenvalues[0,flat_bands]
        if np.allclose(first_energies[0], first_energies, rtol=1e-4, atol=1e-7):
            # Continue; the bands are degenerate amongst each other
            pass
        
        else:
            # Terminate otherwise
            raise NonDegenerateFlatBandsException
        
    def identify_flat_bands(self):
        """
        Solves for the eigenvalues and eigenvectors of the given Hamiltonian
        at linearly spaced samples of k in the FBZ, determined by the specified
        number of intervals in k-space. It will then try to determine the
        existence of flat bands and return them if present. If no flat bands
        exist, the program terminates here.

        Returns
        -------
        None.

        """
        
        self._eigenvalues = []
        # Eigenvectors are used to form the matrix U anyway, so just get them now
        self._eigenvectors = []

        # Calculate eigenvalues and eigenvectors
        for k in self._k_samples:
            sample_hamiltonian = self._hamiltonian(k)
            # Calculate eigenvalues and normalized eigenvectors
            sample_eigenvalues, sample_eigenvectors = LA.eigh(sample_hamiltonian)
            self._eigenvalues.append(sample_eigenvalues)
            self._eigenvectors.append(sample_eigenvectors)
            
        self._eigenvalues = np.array(self._eigenvalues)  # (or eigenenergies, energies, etc.)
        self._eigenvectors = np.array(self._eigenvectors)
        self._N_orb = self._eigenvalues.shape[1]
        
        # Now identify flat bands (if any exist).
        # This algorithm is technically flawed because it depends on
        # np.linalg.eigh returning the eigenvalues of the bands in the
        # same "order" consistently, but it'll probably work so long as bands
        # don't cross.
        
        # Fills with the indices of bands that are flat (enough).
        flat_bands = []
        
        for i in range(self._N_orb):
            energies = self._eigenvalues[:,i]
            first_value = energies[0]
            # Check if all the energies are suitably close to the first value
            if np.allclose(first_value, energies, rtol=1e-4, atol=1e-7):
                # If True, identify as a flat band, otherwise ignore
                flat_bands.append(i)
                
        if (len(flat_bands) != 0):
            print("%.i flat bands detected:" % (len(flat_bands)))
            # List orbital indices of the flat bands
            print(flat_bands)
        
        else:
            # Terminate otherwise—there's no point continuing if there are
            # no flat bands.
            raise NoFlatBandException
            
        # To-do: implement an algorithm that correctly keeps a given band
        # in a given column even if there are intersecting bands.
            
        return flat_bands
    
    def charge_1_spectra(self, flat_bands):
        """
        Calculates the charge +1 excitation spectra for the specified
        degenerate flat bands. Everything should give a flat spectrum at a
        single energy, regardless of the given Hamiltonian. Terminates if
        the given bands are not degenerate.

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
        
        charge_1_energies = []
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,flat_bands]
        
        for i in range(self._N):
            # Find the appropriate matrix of eigenvectors for the given k
            U = reduced_eigenvectors[i]
            # Calculate the matrix, R
            R = 0.5 * self._mod_U * self._epsilon * np.einsum("am,an->mn", np.conjugate(U), U)
            # Now get the eigenenergies
            charge_1_energies.append(LA.eigvalsh(R))
            
        self._charge_1_energies = np.array(charge_1_energies)
        
        # Plot the excitation spectra (should always be flat)
        for i in range(self._charge_1_energies.shape[1]):
            plt.plot(self._k_samples * self._a, self._charge_1_energies[:,i], ".")
            
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
        bands are not degenerate.

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
        
        charge_2_energies = []
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,flat_bands]
        
        # To solve at each pair momentum, p, all the FBZ momenta, k, need to be
        # summed over (pain, I know—it probably gets worse with trions)
        
        # Iterate over each p
        for i in range(self._N):
            h = 0.j
            # Iterate over each k
            for j in range(self._N):
                # See page 30 of the paper for clarification on the process being
                # performed here
                U_k = reduced_eigenvectors[j]
                P_k = U_k @ np.conjugate(U_k.T)
                # The modulo wraps it back into an FBZ wavevector (whose related
                # eigenvectors have already been calculated)
                # Different cases for j corresponding to positive and negative
                # wavevectors are needed because the smallest wavevectors
                # correspond to j around N/2, not 0
                if (j < (self._N - 1)/2):
                    U_p_plus_k = reduced_eigenvectors[(i - (self._N // 2 - j)) % self._N]
                else:
                    U_p_plus_k = reduced_eigenvectors[(i + (self._N // 2 + j)) % self._N]
                    
                P_p_plus_k = U_p_plus_k @ np.conjugate(U_p_plus_k.T)
                h += P_p_plus_k * P_k.T
                
            energies = LA.eigvalsh(h / self._N)
            charge_2_energies.append(self._mod_U * (self._epsilon - energies))
            
        self._charge_2_energies = np.array(charge_2_energies)
        
        # Plot the excitation spectra
        for i in range(self._charge_2_energies.shape[1]):
            plt.plot(self._k_samples * self._a, self._charge_2_energies[:,i], ".")
            
        plt.plot(np.array([-np.pi, np.pi]), self._mod_U*self._epsilon*np.ones(2), "--", color="black", label=r"$\epsilon |U|$")
            
        plt.grid(True)
        plt.legend()
        plt.title("Excitation Spectra", fontsize=18)
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
        
        for i in range(self._eigenvalues.shape[1]):
            plt.plot(self._k_samples * self._a, self._eigenvalues[:,i], ".", label="Band %.i" % (i))
            
        plt.grid(True)
        plt.title("Band Structure", fontsize=18)
        plt.xlabel(r"$ka$")
        plt.ylabel("Energy")
        plt.legend()
        plt.show()
        
    
def dimerized_SSH(k):
    """
    Hamiltonian of the dimerized SSH chain (taken from the paper).

    Parameters
    ----------
    k : float
        Wavenumber to sample at.

    Returns
    -------
    h : ndarray of complex128
        Matrix of the Hamiltonian at the given k.

    """
    
    sigma_2 = np.array([[0., -1.j], [1.j, 0.]])
    sigma_3 = np.array([[1., 0.], [0., -1.]], dtype="complex128")
    
    h = np.sin(k) * sigma_2 + np.cos(k) * sigma_3
    
    return h


test = excitation_solver(1., dimerized_SSH, 1.)