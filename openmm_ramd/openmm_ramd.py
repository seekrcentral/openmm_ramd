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
import openmm_ramd.logger as logger
from openmm_ramd.base import kcal_per_mole_per_angstrom

#import .base
#import .force
#import .logger
#from .base import kcal_per_mole_per_angstrom

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
                 properties=None, log_file_name=None, ramd_force_group=1,
                 ramdSeed=0):
        self.log_file_name = log_file_name
        self.logger = None
        if log_file_name is not None:
            self.logger = logger.RAMD_logger(log_file_name)
            
        self.force_handler = force.RAMD_Force_Handler(ramd_force_magnitude, 
                                                 ligand_atom_indices,
                                                 receptor_atom_indices,
                                                 ramd_force_group)
        self.force_handler.make_RAMD_force_object()
        #self.force_handler.set_new_RAMD_force_vector()
        forcenum = system.addForce(self.force_handler.force_object)
        self.ramdSeed = ramdSeed
        if ramdSeed > 0:
            integrator.setRandomNumberSeed(ramdSeed)
        
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
    
    def check_inputs(self, ramdSteps, forceOutFreq, mdSteps, mdStart, rMinMd):
        # mdSteps has not yet been implemented
        assert mdSteps == 0, "RAMD with MD is not yet implemented: mdSteps "\
            "must be zero."
        #assert ramdSteps % forceOutFreq == 0, "The number of RAMD steps is "\
        #    "not a multiple of 'forceOutFreq'."
        assert mdStart == "no", "RAMD with MD is not yet implemented: mdStart "\
            "must be 'no'."
        assert mdSteps == 0, "RAMD with MD is not yet implemented: rMinMd "\
            "must be zero."
        return
    
    def max_distance_exceeded(self, timestep):
        self.logger.exit_log(
            "{}  > MAX DISTANCE LIGAND COM - PROTEIN COM REACHED".format(
            timestep), print_also=True)
        self.logger.exit_log(
            "{}  > LIGAND EXIT EVENT DETECTED: STOP SIMULATION".format(
            timestep), print_also=True)
        self.logger.exit_log(
            "{}  > EXIT OPENMM".format(
            timestep), print_also=True)
        return
    
    def get_lig_com(self, positions=None):
        if positions is None:
            state = self.context.getState(getPositions = True)
            positions = state.getPositions()
        com = base.get_ligand_com(self.system, positions, 
                                  self.force_handler.ligand_atom_indices)
        return com
        
    def get_rec_com(self, positions=None):
        if positions is None:
            state = self.context.getState(getPositions = True)
            positions = state.getPositions()
        com = base.get_ligand_com(self.system, positions, 
                                  self.force_handler.receptor_atom_indices)
        return com
    
    def run_RAMD_sim(self, max_num_steps=1e8, ramdSteps=50, rMinRamd=0.025, 
                     forceOutFreq=50, mdSteps=0, mdStart="no", rMinMd=0, 
                     maxDist=50, debug_level=0):
        counter = 0
        if self.logger is not None:
            self.logger.log("ramdSteps                {}".format(ramdSteps))
            self.logger.log("openmmVersion            {}".format(
                openmm.version.full_version))
            self.logger.log("forceRAMD                {}".format(
                self.force_handler.random_force_magnitude.value_in_unit(
                    kcal_per_mole_per_angstrom)))
            self.logger.log("rMinRamd                 {}".format(rMinRamd))
            self.logger.log("forceOutFreq             {}".format(forceOutFreq))
            self.logger.log("protAtoms                {}".format(
                self.force_handler.receptor_atom_indices))
            self.logger.log("ligAtoms                 {}".format(
                self.force_handler.ligand_atom_indices))
            self.logger.log("ramdSeed                 {}".format(self.ramdSeed))
            self.logger.log("mdSteps                  {}".format(mdSteps))
            self.logger.log("mdStart                  {}".format(mdStart))
            self.logger.log("rMinMd                   {}".format(rMinMd))
            self.logger.log("maxDist                  {}".format(maxDist))
            self.logger.log("debugLevel               {}".format(debug_level))
        
        self.check_inputs(ramdSteps, forceOutFreq, mdSteps, mdStart, rMinMd)
        lig_com = self.get_lig_com()
        rec_com = self.get_rec_com()
        lig_prot_com_distance = np.linalg.norm(
            lig_com.value_in_unit(unit.angstroms) \
            - rec_com.value_in_unit(unit.angstroms))
        self.recompute_RAMD_force()
        random_direction = self.force_handler.random_vector
        random_direction_magnitude = np.linalg.norm(random_direction)
        if self.logger is not None:
            self.logger.log("Pure RAMD simulation is performed")
            self.logger.log("Atoms subject to the random force are: {}".format(
                self.force_handler.ligand_atom_indices))
            self.logger.timestep_log("***** INITIALIZE RAMD SIMULATION *****", counter)
            self.logger.timestep_log("   >>> minimum travelled distance (A): {}"\
                                     .format(rMinRamd), counter)
            vMin = rMinRamd / ramdSteps
            self.logger.timestep_log("   >>> minimum velocity (A/fs): {}"\
                                     .format(vMin), counter)
            self.logger.timestep_log("   >>> LIGAND COM IS: {}".format(lig_com), counter)
            self.logger.timestep_log("   >>> PROTEIN COM IS: {}".format(rec_com), counter)
            self.logger.timestep_log(
                "   >>> DISTANCE LIGAND COM - PROTEIN COM IS: DIST = {}".format(
                    lig_prot_com_distance), counter)
            self.logger.timestep_log(
                "   >>> INITIAL RANDOM DIRECTION: {} :: ||r|| = {}".format(
                    random_direction, random_direction_magnitude), counter)
            self.logger.timestep_log(
                "***** START WITH {} STEPS OF RAMD SIMULATION *****".format(
                    ramdSteps), counter)
            
        old_lig_com = lig_com
        
        # Do the simulation steps and loop here. These are done every step
        while counter < max_num_steps:
            self.step(ramdSteps)
            state = self.context.getState(getPositions = True)
            positions = state.getPositions()
            lig_com = self.get_lig_com(positions)
            rec_com = self.get_rec_com(positions)
            force_direction = self.force_handler.force_vector
            force_direction_magnitude = np.linalg.norm(
                force_direction.value_in_unit(kcal_per_mole_per_angstrom))
            random_direction = self.force_handler.random_vector
            random_direction_magnitude = np.linalg.norm(random_direction)
            lig_prot_com_distance = np.linalg.norm(
                lig_com.value_in_unit(unit.angstroms) \
                - rec_com.value_in_unit(unit.angstroms))
            lig_walk_distance = np.linalg.norm(
                lig_com.value_in_unit(unit.angstroms) \
                - old_lig_com.value_in_unit(unit.angstroms))
            
            if counter % forceOutFreq == 0:
                if self.logger is not None:
                    self.logger.force_log("> LIGAND COM is: {}".format(lig_com), counter)
                    self.logger.force_log("> PROTEIN COM IS {}".format(rec_com), counter)
                    self.logger.timestep_log(
                        "> EXTERNAL FORCE VECTOR (F): {}; ||F|| = {}".format(
                            force_direction, force_direction_magnitude), counter)
                    self.logger.timestep_log(
                        "> EXTERNAL FORCE DIRECTION (r): {}; ||r|| = {}".format(
                            random_direction, random_direction_magnitude), counter)
            
            # These are done every RAMD evaluation
            if self.logger is not None:
                self.logger.timestep_log(
                    "***** EVALUATE {} RAMD STEPS AT TIMESTEP {} *****".format(
                        ramdSteps, counter), counter)
                self.logger.timestep_log(
                    "   >>> DISTANCE LIGAND COM - PROTEIN COM IS: DIST = {}".format(
                        lig_prot_com_distance), counter)
            
            if lig_walk_distance <= rMinRamd:
                self.recompute_RAMD_force()
                if self.logger is not None:
                    self.logger.timestep_log(
                        "   >>> THE DISTANCE TRAVELLED BY THE LIGAND IS: {} (< {})"\
                        .format(lig_walk_distance, rMinRamd), counter)
                    self.logger.timestep_log(
                        "   >>> CONTINUE WITH {} STEPS OF RAMD SIMULATION".format(
                            ramdSteps), counter)
                    random_direction = self.force_handler.random_vector
                    random_direction_magnitude = np.linalg.norm(random_direction)
                    self.logger.timestep_log(
                        "   >>> CHANGE ACCELERATION DIRECTION TO: {}; ||r|| = {}".format(
                            random_direction, random_direction_magnitude), counter)
            
            else:
                if self.logger is not None:
                    self.logger.timestep_log(
                        "   >>> THE DISTANCE TRAVELLED BY THE LIGAND IS: {} (> {})"\
                        .format(lig_walk_distance, rMinRamd), counter)
                    self.logger.timestep_log(
                        "   >>> CONTINUE WITH {} STEPS OF RAMD SIMULATION".format(
                            ramdSteps), counter)
                    self.logger.timestep_log(
                        "   >>> KEEP PREVIOUS ACCELERATION DIRECTION: {}; ||r|| = {}".format(
                            random_direction, random_direction_magnitude), counter)
        
            counter += ramdSteps
            if lig_prot_com_distance > maxDist:
                self.max_distance_exceeded(counter)
                break
            
            old_lig_com = lig_com
        
        return counter

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    pass
    
