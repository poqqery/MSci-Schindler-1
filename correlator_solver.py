# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 02:37:45 2024

@author: Madhuwrit, Louis
"""

from copy import deepcopy

class delta:
    def __init__(self, delta_type, left, right):
        """
        Kronecker delta function.

        Parameters
        ----------
        delta_type : str
            "momentum", "band" or "spin". Determines what two quantity types
            the Kronecker delta will force to be equal.
        left : str
            Value on the left of the Kronecker delta.
        right : str
            Value on the right of the Kronecker delta.

        Returns
        -------
        None.

        """
        self._d_type = delta_type
        self._left = left
        self._right = right
        
    def swap(self):
        
        self._left, self._right = self._right, self._left
    
class operator:
    def __init__(self, momentum, band, spin):
        """
        Superclass to hold gamma dagger (electron) and gamma (hole) operators.

        Parameters
        ----------
        momentum : str
            Operator crystal momentum.
        band : str
            Operator band index.
        spin : str
            Operator string ("up" or "down").

        Returns
        -------
        None.

        """
        self._momentum = momentum
        self._band = band
        self._spin = spin
    
class gamma_dagger(operator):
    def __init__(self, momentum, band, spin):
        """
        Subclass of the operator class for electron operators.

        Returns
        -------
        None.

        """
        super(gamma_dagger, self).__init__(momentum, band, spin)
        
class gamma(operator):
    def __init__(self, momentum, band, spin):
        """
        Subclass of the operator class for hole operators.

        Returns
        -------
        None.

        """
        super(gamma, self).__init__(momentum, band, spin)
    
class product:
    def __init__(self, components, sign=1):
        """
        Holds a product of any given set of electron/hole operators and delta
        functions. The sign of any product is assumed positive (1) unless
        specified otherwise.

        Parameters
        ----------
        components : list of operator superclass and delta objects
            List of operator and/or delta objects in order from left to right.
        sign : int
            Sign of the product (1 for positive or -1 for negative). Default
            is 1.

        Returns
        -------
        None.

        """
        self._components = components
        self._sign = sign
        
    def invert_sign(self):
        
        self._sign *= -1
        
    def swap(self, index_1, index_2):
        
        self._components[index_1], self._components[index_2] = self._components[index_2], self._components[index_1]
        
class correlator:
    def __init__(self, start_product):
        """
        Correlator object, initially consisting of a single product object that
        is broken down into delta functions.

        Parameters
        ----------
        start_product : list of product object (1 instance)
            Product of initial operators in order from left to right. No delta
            functions should be present.

        Returns
        -------
        None.

        """
        # Store as a multi-type list that is expanded out into a series of
        # Kronecker delta functions; each resultant product is stored as another
        # list of deltas inside the larger list, self._correlator. 
        self._objects = [start_product]
        
    def solve(self):
        """
        Solves a correlator object to return a list of a product of Kronecker
        delta functions. Definitely not the most efficient algorithm, but the
        result only needs to be evaluated once and saved.

        Returns
        -------
        None.

        """
        
        solved = False
        
        while (solved == False):
            for i, product_obj in enumerate(self._objects):
                contains_operator = False
                for component in product_obj._components:
                    if ((type(component) == gamma) or (type(component) == gamma_dagger)):
                        # If true, this object is not purely made of deltas and needs
                        # to be broken down
                        contains_operator = True
                        break
                
                if (contains_operator == True):
                    break
                
            if (contains_operator == False):
                # If no operators are present anywhere, the correlator has been
                # solved—exit the function and return the result.
                solved = True
                
            else:
                # Get the product identified to contain an operator (it is
                # guaranteed to contain at least one gamma and gamma dagger);
                # it is already in memory at this point under the index i and
                # the variable product_obj.
                
                # Find the right-most gamma operator (loop backwards)
                for j, component in enumerate(product_obj._components[::-1]):
                    if (type(component) == gamma):
                        if (j == 1):
                            # If j = 1, the hole operator is second to the right;
                            # swapping will annihilate the remaining product.
                            # Insert delta functions to the left of the product
                            gamma_operator = product_obj._components[-2]
                            gamma_dagger_operator = product_obj._components[-1]
                            product_obj._components.insert(0, delta("momentum", gamma_operator._momentum,\
                                gamma_dagger_operator._momentum))
                            product_obj._components.insert(0, delta("band", gamma_operator._band,\
                                gamma_dagger_operator._band))
                            product_obj._components.insert(0, delta("spin", gamma_operator._spin,\
                                gamma_dagger_operator._spin))
                                
                            # Delete the two rightmost operators (gamma and gamma dagger)
                            del product_obj._components[-1]
                            del product_obj._components[-1]
                            
                        else:
                            # If the gamma is not second to the right, another
                            # product needs to be created with the two relevant
                            # operators swapped.
                            dupe_product = deepcopy(product_obj)
                            dupe_product.swap(-j-1, -j)
                            dupe_product.invert_sign()
                            
                            # Insert delta functions to the left of the original product
                            gamma_operator = product_obj._components[-j-1]
                            gamma_dagger_operator = product_obj._components[-j]
                            product_obj._components.insert(0, delta("momentum", gamma_operator._momentum,\
                                gamma_dagger_operator._momentum))
                            product_obj._components.insert(0, delta("band", gamma_operator._band,\
                                gamma_dagger_operator._band))
                            product_obj._components.insert(0, delta("spin", gamma_operator._spin,\
                                gamma_dagger_operator._spin))
                            # Delete the two operators from this product
                            del product_obj._components[-j-1]
                            del product_obj._components[-j]
                            
                            # Add the duplicated correlator with changed properties
                            # to the correlator object list
                            self._objects.insert(i, dupe_product)
                            
                        break
                    
    def clean_result(self):
        """
        Cleans up some redundant delta functions to do with spin.
        self.solve() must be executed first.

        Returns
        -------
        None.

        """
        
        # Delete delta functions that have the same argument on both sides
        for product_obj in self._objects:
            keep_list = []
            for delta in product_obj._components:
                if (delta._d_type == "spin"):
                    if (delta._left == delta._right):
                        # Delta function is redundant; delete it
                        keep_list.append(False)
                        
                    else:
                        keep_list.append(True)
                        
                else:
                    keep_list.append(True)
                    
            duplicate = [product_obj._components[i] for i in range(len(product_obj._components)) if (keep_list[i] == True)]
            product_obj._components = duplicate
            
    def print_result(self):
        """
        Print the result in a way that can (at least a bit) be read by a human.

        Returns
        -------
        None.

        """
        
        for i, product_obj in enumerate(self._objects):
            print("Term %.i:" % (i+1))
            print("\t Sign: %.i \n" % (product_obj._sign))
            for delta in product_obj._components:
                print("\t Delta: " + delta._left + ", " + delta._right)
                    
#%%

operator_1 = gamma("-q3", "j_t", "s'''")
operator_2 = gamma("-q2", "l_t", "s''")
operator_3 = gamma("-q1", "n_t", "s'")
operator_4 = gamma("p+q1+q2+q3", "m_t", "s")
operator_5 = gamma_dagger("p+k1+k2+k3", "m", "s")
operator_6 = gamma_dagger("-k1", "n", "s'")
operator_7 = gamma_dagger("-k2", "l", "s''")
operator_8 = gamma_dagger("-k3", "j", "s'''")
start_product = product([operator_1, operator_2, operator_3, operator_4, operator_5, operator_6, operator_7, operator_8])

four_correlator = correlator(start_product)
four_correlator.solve()
four_correlator.clean_result()
four_correlator.print_result()