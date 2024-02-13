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
            
    def four_electrons_sparse(self, flat_bands, spins, p, num_eigenvalues=10):
        # Epsilon from the UPC can now be assigned
        N_f = len(flat_bands)
        epsilon = N_f / 2.
        
        # Remove the columns of eigenvectors that do not correspond to the flat bands
        reduced_eigenvectors = self._eigenvectors[:,:,:,flat_bands]
        
        # Define the three spins (0 maps to 1, 1 maps to -1 for up and down respectively)
        # sigma = index 0; sigma prime = index 1; sigma double prime = index 2
        spins = np.array(spins)
        s = -2*spins + 1
        
        R_dimensionality = self._N**3
        
        q = np.arange(self._N)
        q=np.broadcast_to(q,(self._N, self._N, self._N))
        q_1 = np.array([[i]*self._N**2 for i in range(self._N)]).flatten()
        q_2 = np.einsum("ijk->ikj",q).flatten()
        q_3 = q.flatten()
        
        # Assume one flat band for now
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
                                    values.append(2. * epsilon)
                                            
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
        four_electrons_energies = sparse.linalg.eigsh(R, k=num_eigenvalues, which="SA", return_eigenvectors=False)
        
        print("p = %.i complete" % (p))
        
        # Change file location to wherever is needed
        np.savetxt("Quarton Multiprocess Energies/%.i_energies.csv" % (p), four_electrons_energies)
            
sigma_1 = np.array([[0., 1.], [1., 0.]], dtype="complex128")
sigma_2 = np.array([[0., -1.j], [1.j, 0.]])
sigma_3 = np.array([[1., 0.], [0., -1.]], dtype="complex128")

def dimerized_SSH(k, spin):
    """
    Hamiltonian of the dimerized-limit SSH chain.

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

def execute_p(solver, p, spins, num_eigenvalues):
    for p_value in p:
        solver.four_electrons_sparse([0], spins, p_value, num_eigenvalues=num_eigenvalues)
            
if (__name__ == "__main__"):
    N_cells = 20
    spins = [0, 0, 1, 1]
    solver = excitation_solver(1., dimerized_SSH, 1., N=N_cells)
    
    processes = []
    
    for p in range(N_cells):
        process = Process(target=execute_p, args=(solver, [p], spins, 10))
        process.start()
        processes.append(process)
        
    for process in processes:
        process.join()
    
    k = np.linspace(0., 2.*np.pi, N_cells + 1)[:-1]
    for p, k_value in enumerate(k):
        energies = np.loadtxt("Quarton Multiprocess Energies/%.i_energies.csv" % (p))
        length = len(energies)
        plt.plot(np.ones(length)*k_value, energies, ".", ms=5, color="red")
    
    plt.grid(True)
    plt.legend()
    plt.title("Quarton (Electrons) Excitation Spectrum", fontsize=18)
    plt.xlabel(r"$pa$")
    plt.ylabel("Energy")
    plt.show()