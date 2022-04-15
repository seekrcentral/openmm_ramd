"""
Test force objects and functions
"""

import pytest
import numpy as np
try:
    import openmm.unit as unit
except ImportError:
    import simtk.openmm.app as openmm_app

import openmm_ramd.openmm_ramd as openmm_ramd
import openmm_ramd.force as force
from openmm_ramd.force import kcal_per_mole_per_angstrom

def test_force_handler_init_and_assign():
    """
    
    """
    ligand_atom_indices = [4,5,6]
    receptor_atom_indices = [0,1,2,3]
    random_force_magnitude = 13.0*kcal_per_mole_per_angstrom
    force_handler = force.RAMD_Force_Handler(
        random_force_magnitude, ligand_atom_indices, receptor_atom_indices, 1)
    force_handler.make_RAMD_force_object()
    result0 = force_handler.force_object.getBondParameters(0)
    assert result0[1][0] == 0.0
    assert result0[1][1] == 0.0
    assert result0[1][2] == 0.0
    
    force_handler.set_new_RAMD_force_vector()
    result1 = force_handler.force_object.getBondParameters(0)
    norm1 = np.linalg.norm(result1[1])
    force_handler.set_new_RAMD_force_vector()
    result2 = force_handler.force_object.getBondParameters(0)
    norm2 = np.linalg.norm(result2[1])
    assert np.isclose(norm1, norm2)
    assert not np.isclose(result1[1], result2[1]).all()
    
    return