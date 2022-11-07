"""
force.py

Set the customRamdForce and other force subclasses.
"""

import numpy as np
try:
    import openmm
except ImportError:
    import simtk.openmm as openmm
    
try:
    import openmm.app as openmm_app
except ImportError:
    import simtk.openmm.app as openmm_app
    
try:
    import openmm.unit as unit
except ImportError:
    import simtk.openmm.unit as unit
    
from openmm_ramd.base import kcal_per_mole_per_angstrom

RAMD_FORCE_EXPRESSION = "-(f_x*x1 + f_y*y1 + f_z*z1)"

class RAMD_Force_Handler():
    """
    An object to store the OpenMM Force() object for a RAMD force.
    """
    def __init__(self, random_force_magnitude, ligand_atom_indices, 
                 receptor_atom_indices, force_group):
        self.random_force_magnitude = random_force_magnitude
        self.ligand_atom_indices = ligand_atom_indices
        self.receptor_atom_indices = receptor_atom_indices
        self.force_group = force_group
        self.force_object = None
        self.bond_index = None
        self.group_index_lig = None
        self.group_index_rec = None
        self.force_vector = None
        self.random_vector = None
        return

    def make_RAMD_force_object(self):
        """
        Make a Force object that can be used in OpenMM that acts as the
        RAMD external force.
        """
        
        if self.receptor_atom_indices is None \
                or len(self.receptor_atom_indices) == 0:
            self.receptor_atom_indices = None
            force = openmm.CustomCentroidBondForce(1, RAMD_FORCE_EXPRESSION)
            force.setForceGroup(self.force_group)
            self.group_index_lig = force.addGroup(self.ligand_atom_indices)
            force.addPerBondParameter("f_x")
            force.addPerBondParameter("f_y")
            force.addPerBondParameter("f_z")
            self.bond_index = force.addBond([self.group_index_lig], 
                                            [0.0*kcal_per_mole_per_angstrom, 
                                             0.0*kcal_per_mole_per_angstrom, 
                                             0.0*kcal_per_mole_per_angstrom])
            
        else:
            force = openmm.CustomCentroidBondForce(2, RAMD_FORCE_EXPRESSION)
            force.setForceGroup(self.force_group)
            self.group_index_lig = force.addGroup(self.ligand_atom_indices)
            self.group_index_rec = force.addGroup(self.receptor_atom_indices)
            force.addPerBondParameter("f_x")
            force.addPerBondParameter("f_y")
            force.addPerBondParameter("f_z")
            self.bond_index = force.addBond([self.group_index_lig, 
                                             self.group_index_rec], 
                                            [0.0*kcal_per_mole_per_angstrom, 
                                             0.0*kcal_per_mole_per_angstrom, 
                                             0.0*kcal_per_mole_per_angstrom])
        self.force_object = force
        self.force_vector = np.array([0.0*kcal_per_mole_per_angstrom, 
                                      0.0*kcal_per_mole_per_angstrom, 
                                      0.0*kcal_per_mole_per_angstrom])
        self.random_vector = np.array([0.0, 0.0, 0.0])
        return
    
    def set_new_RAMD_force_vector(self):
        """
        Modify the force object to have a new force vector
        """
        f_gauss_x = np.random.normal()
        f_gauss_y = np.random.normal()
        f_gauss_z = np.random.normal()
        f_gauss_len = np.sqrt(f_gauss_x**2 + f_gauss_y**2 + f_gauss_z**2)
        force_vector_unscaled = np.array(
            [f_gauss_x/f_gauss_len, f_gauss_y/f_gauss_len, 
             f_gauss_z/f_gauss_len])
        force_vector = force_vector_unscaled * self.random_force_magnitude
        f_x = force_vector[0]
        f_y = force_vector[1]
        f_z = force_vector[2]
        force = self.force_object
        if self.receptor_atom_indices is None \
                or len(self.receptor_atom_indices) == 0:
            force.setBondParameters(self.bond_index, [self.group_index_lig], 
                                    [f_x, f_y, f_z])
        else:
            force.setBondParameters(self.bond_index, [self.group_index_lig, 
                                                      self.group_index_rec], 
                                    [f_x, f_y, f_z])
        self.force_vector = force_vector
        self.random_vector = force_vector_unscaled
        return
    