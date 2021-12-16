"""
Provide the primary functions.
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

import openmm_ramd.base as base
import openmm_ramd.force as force
from openmm_ramd.base import kcal_per_mole_per_angstrom

class RAMDSimulation(openmm_app.Simulation):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """
    def __init__(self, topology, system, integrator, ramd_force_magnitude, 
                 ligand_atom_indices, receptor_atom_indices, platform=None, 
                 properties=None, log_file_name=None, ramd_force_group=1):
        self.log_file_name = log_file_name
        self.force_handler = force.RAMD_Force_Handler(ramd_force_magnitude, 
                                                 ligand_atom_indices,
                                                 receptor_atom_indices,
                                                 ramd_force_group)
        self.force_handler.make_RAMD_force_object()
        #self.force_handler.set_new_RAMD_force_vector()
        forcenum = system.addForce(self.force_handler.force_object)
        
        if platform is None:
            assert properties is None, "If platform is not set, properties cannot "\
                "be set either."
            #simulation = openmm_app.Simulation(topology, system, integrator)
            super(RAMDSimulation, self).__init__(
                topology, system, integrator)
        else:
            assert properties is not None, "If platform is set, properties must "\
                "also be set."
            #simulation = openmm_app.Simulation(topology, system, integrator, 
            #                                   platform, properties)
            super(RAMDSimulation, self).__init__(
                topology, system, integrator, platform, properties)
            
    def recompute_RAMD_force(self):
        self.force_handler.set_new_RAMD_force_vector()
        self.force_handler.force_object.updateParametersInContext(self.context)

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    pass
    
