"""
base.py

Contain base classes, constants, and other objects used for RAMD.
"""

import numpy as np
try:
    import openmm.unit as unit
except ImportError:
    import simtk.openmm.unit as unit
    
kcal_per_mole_per_angstrom = unit.kilocalories_per_mole / unit.angstrom

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
    

"""
def get_site_ligand_distance(topology, group1, group2):
    ""
    Compute the distance between the centers of masses of two groups of
    atom indices (group1 and group2) in the pdb_filename structure.
    ""
    traj = mdtraj.load(pdb_filename)
    traj1 = traj.atom_slice(group1)
    traj2 = traj.atom_slice(group2)
    com1_array = mdtraj.compute_center_of_mass(traj1)
    com2_array = mdtraj.compute_center_of_mass(traj2)
    com1 = com1_array[0,:]
    com2 = com2_array[0,:]
    distance = np.linalg.norm(com2-com1)
    return distance * unit.nanometer
"""