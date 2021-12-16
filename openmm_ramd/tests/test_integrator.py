"""
Test integrator objects and functions
"""

import pytest
try:
    import openmm.unit as unit
except ImportError:
    import simtk.openmm.app as openmm_app

import openmm_ramd
from openmm_ramd.integrator import kcal_per_mole_per_angstrom

def test_customRamdIntegrator_set_get():
    """
    
    """
    """
    test_integrator = openmm_ramd.integrator.CustomRamdIntegrator()
    assert test_integrator.getRAMDForceMagnitude() == 0.0*kcal_per_mole_per_angstrom
    
    random_force_magnitude = 13.0*kcal_per_mole_per_angstrom
    test_integrator.setRAMDForceMagnitude(random_force_magnitude)
    assert random_force_magnitude == test_integrator.getRAMDForceMagnitude()
    """