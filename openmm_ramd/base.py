"""
base.py

Contain base classes, constants, and other objects used for RAMD.
"""

import numpy as np
import mdtraj
try:
    import openmm.unit as unit
except ImportError:
    import simtk.openmm.unit as unit
    
kcal_per_mole_per_angstrom = unit.kilocalories_per_mole / unit.angstrom

def get_ligand_mass(system, group1):
    """
    Compute the mass of a group of atoms given an OpenMM system
    and the indices of the atoms of interest.
    
    Parameters:
    -----------
    system : System()
        The OpenMM System object which contains the atomic masses.
        
    group1 : list
        A list of integers representing atom indices of the group
        the center of mass is being computed for.
        
    Returns:
    --------
    mass : Quantity
        The mass of the group of atoms
    """
    total_mass = 0.0 * unit.daltons
    for index in group1:
        atom_mass = system.getParticleMass(index)
        total_mass += atom_mass
    
    return total_mass

def get_ligand_com(system, positions, group1):
    """
    Compute the center of mass of a group of atoms given an OpenMM
    system, the system's atomic positions, and the indices of the
    atoms of interest.
    
    Parameters:
    -----------
    system : System()
        The OpenMM System object which contains the atomic masses.
        
    positions : Quantity
        A n-by-3 array of positions of all atoms in the system. Can
        be the output of State.getPositions()
        
    group1 : list
        A list of integers representing atom indices of the group
        the center of mass is being computed for.
        
    Returns:
    --------
    com : Quantity
        A 1x3 array representing the x,y,z coordinates of the center of
        mass of the group of atoms.
    """
    total_mass = 0.0 * unit.daltons
    com = np.array([0, 0, 0]) * unit.nanometers * unit.daltons
    for index in group1:
        atom_mass = system.getParticleMass(index)
        total_mass += atom_mass
        com += atom_mass * positions[index]
        
    com /= total_mass
    return com

def get_ligand_receptor_distance(system, positions, group1, group2):
    total_mass1 = 0.0 * unit.daltons
    com1 = np.array([0, 0, 0]) * unit.nanometers * unit.daltons
    for index in group1:
        atom_mass = system.getParticleMass(index)
        total_mass1 += atom_mass
        com1 += atom_mass * positions[index]
        
    com1 /= total_mass1
    total_mass2 = 0.0 * unit.daltons
    com2 = np.array([0, 0, 0]) * unit.nanometers * unit.daltons
    for index in group2:
        atom_mass = system.getParticleMass(index)
        total_mass2 += atom_mass
        com2 += atom_mass * positions[index]
        
    com2 /= total_mass2
    distance = np.linalg.norm(com2.value_in_unit(unit.nanometer)\
                              -com1.value_in_unit(unit.nanometer))
    return distance*unit.nanometer

""" # TODO: marked for removal
def align_positions_by_reference(positions, ref_positions, selection):
    # Make a Traj object from an OpenMM system
    
    traj.superpose(traj, atom_indices=selection, ref_atom_indices=selection)
"""

def initialize_indices(indices):
    """
    Break down a string list of integers into a Python list of ints.
    """
    assert indices != ""
    integers = indices.split(",")
    for integer in integers:
        integer_int = int(integer)
    
    return list(map(int, integers))

def get_indices_ligand(pdb_filename, lig_resname):

    """
    Return the atom indices of the ligand molecule (beginning from 0).

    Parameters
    ----------
    pdb_filename: str
        User-defined PDB file.

    lig_resname: str
        Three-letter name for the ligand residue.

    """
    traj = mdtraj.load(pdb_filename)
    atom_indices = traj.topology.select(f"resname {lig_resname}")
    return atom_indices
