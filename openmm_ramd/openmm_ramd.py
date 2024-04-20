"""
Provide the primary functions.
"""

import numpy as np
import mdtraj
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
                 ligand_atom_indices, receptor_atom_indices=None, 
                 ramd_steps=50, r_min_ramd=0.025, 
                 force_out_freq=50, md_steps=0, md_start="no", r_min_md=0, 
                 max_dist=50, debug_level=0, platform=None, 
                 properties=None, log_file_name=None, ramd_force_group=1,
                 ramd_seed=0, align_receptor=False, position_print_freq=None):
        self.log_file_name = log_file_name
        self.logger = None
        if log_file_name is not None:
            self.logger = logger.RAMD_logger(log_file_name)
            
        self.force_handler = force.RAMD_Force_Handler(ramd_force_magnitude, 
                                                 ligand_atom_indices,
                                                 receptor_atom_indices,
                                                 ramd_force_group)
        self.force_handler.make_RAMD_force_object()
        forcenum = system.addForce(self.force_handler.force_object)
        self.ramd_seed = ramd_seed
        self.old_lig_com = None
        self.counter = 0
        self.ramd_steps = ramd_steps
        self.r_min_ramd = r_min_ramd
        self.force_out_freq = force_out_freq
        if position_print_freq == None:
            self.position_print_freq = self.force_out_freq
        else:
            self.position_print_freq = position_print_freq
            
        self.md_steps = md_steps
        self.md_start = md_start
        self.r_min_md = r_min_md
        self.max_dist = max_dist
        self.debug_level = debug_level
        if ramd_seed > 0:
            integrator.setRandomNumberSeed(ramd_seed)
        if align_receptor:
            self.mdtraj_topology = mdtraj.Topology.from_openmm(topology)
            self.mdtraj_trajectory = None
            self.mdtraj_topology_ref = mdtraj.Topology.from_openmm(topology)
            self.mdtraj_trajectory_ref = None
        else:
            self.mdtraj_topology = None
            self.mdtraj_trajectory = None
            self.mdtraj_topology_ref = None
            self.mdtraj_trajectory_ref = None
            
        self.ref_positions = None
        if receptor_atom_indices is None:
            align_receptor = False
        self.align_receptor = align_receptor
        
        if platform is None:
            assert properties is None, "If platform is not set, properties "\
                "cannot be set either."
            #simulation = openmm_app.Simulation(topology, system, integrator)
            super(RAMDSimulation, self).__init__(
                topology, system, integrator)
        else:
            assert properties is not None, "If platform is set, properties "\
                "must also be set."
            #simulation = openmm_app.Simulation(topology, system, integrator, 
            #                                   platform, properties)
            super(RAMDSimulation, self).__init__(
                topology, system, integrator, platform, properties)
    
    def recompute_RAMD_force(self):
        print("recomputing force!")
        self.force_handler.set_new_RAMD_force_vector()
        self.force_handler.force_object.updateParametersInContext(self.context)
    
    def check_inputs(self):
        # md_steps has not yet been implemented
        assert self.md_steps == 0, "RAMD with MD is not yet implemented: "\
            "mdSteps must be zero."
        assert self.md_start == "no", "RAMD with MD is not yet implemented: "\
            "mdStart must be 'no'."
        assert self.r_min_md == 0, "RAMD with MD is not yet implemented: "\
            "rMinMd must be zero."
        assert self.force_out_freq % self.ramd_steps == 0, \
            "ForceOutFreq must be a multiple of ramdSteps."
        assert self.position_print_freq % self.ramd_steps == 0, \
            "PositionPrintFreq must be a multiple of ramdSteps."
        return
    
    def max_distance_exceeded(self, timestep):
        self.logger.exit_log(
            "{}  > MAX DISTANCE LIGAND COM - PROTEIN COM REACHED".format(
            timestep), print_also=False)
        self.logger.exit_log(
            "{}  > LIGAND EXIT EVENT DETECTED: STOP SIMULATION".format(
            timestep), print_also=False)
        self.logger.exit_log(
            "{}  > EXIT OPENMM".format(
            timestep), print_also=False)
        return
    
    def max_time_exceeded(self, timestep):
        self.logger.exit_log(
            "{}  > MAXIMUM TIME EXCEEDED".format(
            timestep), print_also=False)
        self.logger.exit_log(
            "{}  > EXIT OPENMM".format(
            timestep), print_also=False)
        return
    
    def get_lig_mass(self):
        mass = base.get_ligand_mass(
            self.system, self.force_handler.ligand_atom_indices)
        return mass.value_in_unit(unit.daltons)
    
    def set_reference_positions(self, positions=None):
        if positions is None:
            state = self.context.getState(getPositions = True)
            positions = state.getPositions()
        xyz = np.array(positions.value_in_unit(unit.nanometers))
        self.mdtraj_trajectory_ref = mdtraj.Trajectory(
                    xyz=xyz, topology=self.mdtraj_topology_ref)
        self.mdtraj_trajectory = mdtraj.Trajectory(
                    xyz=xyz, topology=self.mdtraj_topology)
        return
        
    def align_positions_by_reference(self, positions, ref_indices):
        xyz = np.array(positions.value_in_unit(unit.nanometers))
        self.mdtraj_trajectory.xyz = xyz
        self.mdtraj_trajectory.superpose(
            self.mdtraj_trajectory_ref, atom_indices=ref_indices, 
            ref_atom_indices=ref_indices)
        
        # DEBUG
        #pdb_filename = f"/home/lvotapka/tmp/ramd_test{self.counter}.pdb"
        #self.mdtraj_trajectory.save(pdb_filename)
        return
    
    def get_lig_com(self, positions=None, skip_alignment=False):
        if positions is None:
            state = self.context.getState(getPositions = True)
            positions = state.getPositions()
        
        if self.align_receptor and not skip_alignment:
            # Align by receptor positions
            self.align_positions_by_reference(
                positions, self.force_handler.receptor_atom_indices)
            positions = self.mdtraj_trajectory.xyz[0,:,:] * unit.nanometer
        
        com = base.get_ligand_com(self.system, positions, 
                                  self.force_handler.ligand_atom_indices)
        return com
        
    def get_rec_com(self, positions=None, skip_alignment=False):
        if positions is None:
            state = self.context.getState(getPositions = True)
            positions = state.getPositions()
        
        if self.align_receptor and not skip_alignment:
            # Align by receptor positions
            self.align_positions_by_reference(
                positions, self.force_handler.receptor_atom_indices)
            positions = self.mdtraj_trajectory.xyz[0,:,:] * unit.nanometer
        
        com = base.get_ligand_com(self.system, positions, 
                                  self.force_handler.receptor_atom_indices)
        return com
    
    def RAMD_start(self):
        self.counter = 0
        if self.logger is not None:
            self.logger.log("ramdSteps                {}".format(
                self.ramd_steps))
            self.logger.log("openmmVersion            {}".format(
                openmm.version.full_version))
            #self.logger.log("forceRAMD                {}".format(
            self.logger.log("forceRAMD                {} kcal/(mol * A)".format(
                self.force_handler.random_force_magnitude.value_in_unit(
                    kcal_per_mole_per_angstrom)))
            self.logger.log("rMinRamd                 {} A".format(
                self.r_min_ramd.value_in_unit(unit.angstroms)))
            self.logger.log("forceOutFreq             {}".format(
                self.force_out_freq))
            self.logger.log("positionPrintFreq        {}".format(
                self.position_print_freq))
            self.logger.log("timeStep                 {}".format(
                self.integrator.getStepSize()))
            self.logger.log("temperature              {}".format(
                self.integrator.getTemperature()))
            self.logger.log("protAtoms                {}".format(
                self.force_handler.receptor_atom_indices))
            self.logger.log("ligAtoms                 {}".format(
                self.force_handler.ligand_atom_indices))
            self.logger.log("ramdSeed                 {}".format(
                self.ramd_seed))
            self.logger.log("mdSteps                  {}".format(self.md_steps))
            self.logger.log("mdStart                  {}".format(self.md_start))
            self.logger.log("rMinMd                   {}".format(self.r_min_md))
            self.logger.log("maxDist                  {} A".format(
                self.max_dist.value_in_unit(unit.angstroms)))
            self.logger.log("debugLevel               {}".format(
                self.debug_level))
        
        self.check_inputs()
        if self.force_handler.receptor_atom_indices is not None\
                and self.align_receptor:
            self.set_reference_positions()
            
        lig_com = self.get_lig_com()
        lig_mass = self.get_lig_mass()
        if self.force_handler.receptor_atom_indices is not None:
            if not self.align_receptor:
                assert not self.align_receptor, \
                    "align_receptor must be set to False if no receptor is "\
                    "defined."
            
            rec_com = self.get_rec_com()
            lig_prot_com_distance = np.linalg.norm(
                lig_com.value_in_unit(unit.angstroms) \
                - rec_com.value_in_unit(unit.angstroms))
        self.recompute_RAMD_force()
        random_direction = self.force_handler.random_vector
        random_direction_magnitude = np.linalg.norm(random_direction)
        if self.logger is not None:
            self.logger.log("Pure RAMD simulation is performed")
            lig_indices_str = " ".join(
                map(str, self.force_handler.ligand_atom_indices))
            self.logger.log("Atoms subject to the random force are: {}".format(
                lig_indices_str))
            self.logger.log(f"calculated ligand mass {lig_mass}")
            self.logger.timestep_log("***** INITIALIZE RAMD SIMULATION *****", 
                                     self.counter)
            self.logger.timestep_log(
                "   >>> minimum travelled distance (A): {}".format(
                    self.r_min_ramd.value_in_unit(unit.angstroms)), 
                                     self.counter)
            vMin = self.r_min_ramd / self.ramd_steps
            self.logger.timestep_log(
                "   >>> minimum velocity (A/fs): {}".format(
                    vMin.value_in_unit(unit.angstroms)), self.counter)
            self.logger.timestep_log("   >>> LIGAND COM IS: {} A".format(
                lig_com.value_in_unit(unit.angstroms)), self.counter)
            if self.force_handler.receptor_atom_indices is not None:
                self.logger.timestep_log("   >>> PROTEIN COM IS: {} A".format(
                    rec_com.value_in_unit(unit.angstroms)), self.counter)
                self.logger.timestep_log(
                    "   >>> DISTANCE LIGAND COM - PROTEIN COM IS: DIST = {} A".\
                    format(lig_prot_com_distance), self.counter)
            self.logger.timestep_log(
                "   >>> INITIAL RANDOM DIRECTION: {} :: ||r|| = {}".format(
                    random_direction, random_direction_magnitude), self.counter)
            self.logger.timestep_log(
                "***** START WITH {} STEPS OF RAMD SIMULATION *****".format(
                    self.ramd_steps), self.counter)
        
        self.old_lig_com = lig_com
        return lig_com
    
    def RAMD_step(self, numSteps=50):
        self.step(numSteps)
        # Obtain positions for either a possible force update or position print
        if (self.counter % self.force_out_freq == 0) \
                or (self.counter % self.position_print_freq == 0):
            state = self.context.getState(getPositions = True)
            positions = state.getPositions()
            lig_com = self.get_lig_com(positions)
            if self.force_handler.receptor_atom_indices is not None:
                rec_com = self.get_rec_com(positions, skip_alignment=True)
                lig_prot_com_distance = np.linalg.norm(
                    lig_com.value_in_unit(unit.angstroms) \
                    - rec_com.value_in_unit(unit.angstroms))
            else:
                lig_prot_com_distance = np.linalg.norm(
                    lig_com.value_in_unit(unit.angstroms))
        
        if self.counter % self.position_print_freq == 0:
            if self.logger is not None:
                self.logger.force_log("> LIGAND COM IS: {} A".format(
                    lig_com.value_in_unit(unit.angstroms)), self.counter)
                if self.force_handler.receptor_atom_indices is not None:
                    self.logger.force_log("> PROTEIN COM IS: {} A".format(
                        rec_com.value_in_unit(unit.angstroms)), self.counter)
        
        if (self.counter % self.force_out_freq == 0):
            force_direction = self.force_handler.force_vector
            force_direction_magnitude = np.linalg.norm(
                force_direction.value_in_unit(kcal_per_mole_per_angstrom))
            random_direction = self.force_handler.random_vector
            random_direction_magnitude = np.linalg.norm(random_direction)
            
            lig_walk_distance = np.linalg.norm(
                lig_com.value_in_unit(unit.angstroms) \
                - self.old_lig_com.value_in_unit(unit.angstroms)) * unit.angstroms
            
            self.old_lig_com = lig_com
            
            if self.logger is not None:
                self.logger.timestep_log(
                    "> EXTERNAL FORCE VECTOR (F): {}; ||F|| = {}".format(
                        force_direction, force_direction_magnitude), 
                                         self.counter)
                self.logger.timestep_log(
                    "> EXTERNAL FORCE DIRECTION (r): {}; ||r|| = {}".format(
                        random_direction, random_direction_magnitude), 
                                         self.counter)
            
            if lig_walk_distance <= self.r_min_ramd:
                self.recompute_RAMD_force()
                if self.logger is not None:
                    self.logger.timestep_log(
                        "   >>> THE DISTANCE TRAVELLED BY THE LIGAND IS: {} (< {}) A"\
                        .format(lig_walk_distance, self.r_min_ramd), self.counter)
                    self.logger.timestep_log(
                        "   >>> CONTINUE WITH {} STEPS OF RAMD SIMULATION".format(
                            self.ramd_steps), self.counter)
                    random_direction = self.force_handler.random_vector
                    random_direction_magnitude = np.linalg.norm(random_direction)
                    self.logger.timestep_log(
                        "   >>> CHANGE ACCELERATION DIRECTION TO: {}; ||r|| = {}"\
                            .format(random_direction, random_direction_magnitude), 
                            self.counter)
            
            else:
                if self.logger is not None:
                    self.logger.timestep_log(
                        "   >>> THE DISTANCE TRAVELLED BY THE LIGAND IS: {} (> {})"\
                        .format(lig_walk_distance, self.r_min_ramd), self.counter)
                    self.logger.timestep_log(
                        "   >>> CONTINUE WITH {} STEPS OF RAMD SIMULATION".format(
                            self.ramd_steps), self.counter)
                    self.logger.timestep_log(
                        "   >>> KEEP PREVIOUS ACCELERATION DIRECTION: {}; ||r|| = {}".format(
                            random_direction, random_direction_magnitude), self.counter)
        
        # Unnecessary output
        # These are done every RAMD evaluation
        #if self.logger is not None:
        #    self.logger.timestep_log(
        #        "***** EVALUATE {} RAMD STEPS AT TIMESTEP {} *****".format(
        #            self.ramd_steps, self.counter), self.counter)
        #    if self.force_handler.receptor_atom_indices is not None:
        #        self.logger.timestep_log(
        #            "   >>> DISTANCE LIGAND COM - PROTEIN COM IS: DIST = {} A"\
        #                .format(lig_prot_com_distance), self.counter)
        
        self.counter += self.ramd_steps
        return lig_prot_com_distance
    
    def run_RAMD_sim(self, max_num_steps=1e8):
        self.counter = 0
        self.context.setTime(0.0)
        start_lig_prot_dist = self.RAMD_start()
        # Do the simulation steps and loop here. These are done every step
        successful_escape = False
        while self.counter < max_num_steps:
            lig_prot_distance = self.RAMD_step(self.ramd_steps)
            if lig_prot_distance * unit.angstroms > self.max_dist:
                self.max_distance_exceeded(self.counter)
                successful_escape = True
                break
        
        if not successful_escape:
            # Then we ran out of time
            self.max_time_exceeded(max_num_steps)
            
        return self.counter

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    pass
    
