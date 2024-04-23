"""
integrator.py

Define the integrator that will be used for fast calculation and retrieval
of ligand/receptor COM-COM distance.
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

class RamdIntegrator(openmm.CustomIntegrator):
    def __init__(self, dt, lig_indices, rec_indices, starting_positions):
        # Initialize the customIntegrator
        super(RamdIntegrator, self).__init__(dt)
        self.lig_indices = lig_indices
        self.rec_indices = rec_indices
        self.starting_positions = starting_positions
        if self.rec_indices is None:
            self.doing_rec = False
        else:
            self.doing_rec = True
        
        self.prepare_integration_quantities()
        self.prepare_com_com_calc()
        self.perform_first_step_procedures()
        self.perform_integration_steps()
        self.perform_com_com_calculation()
        return
    
    def prepare_com_com_calc(self):
        # Add global variables for COM-COM calc
        self.addGlobalVariable("step_count", 0)
        self.addGlobalVariable("lig_mass", 0)
        self.addGlobalVariable("lig_weighted_x", 0)
        self.addGlobalVariable("lig_weighted_y", 0)
        self.addGlobalVariable("lig_weighted_z", 0)
        if self.doing_rec:
            self.addGlobalVariable("rec_mass", 0)
            self.addGlobalVariable("rec_weighted_x", 0)
            self.addGlobalVariable("rec_weighted_y", 0)
            self.addGlobalVariable("rec_weighted_z", 0)
        self.addGlobalVariable("eps", 0.001 * unit.nanometers)
        for i, lig_index in enumerate(self.lig_indices):
            lig_position = self.starting_positions[lig_index]
            print(f"Adding lig position {i}:", lig_position)
            self.addGlobalVariable(f"lig_x{i}", lig_position[0])
            self.addGlobalVariable(f"lig_y{i}", lig_position[1])
            self.addGlobalVariable(f"lig_z{i}", lig_position[2])
        
        if self.doing_rec:
            for i, rec_index in enumerate(self.rec_indices):
                rec_position = self.starting_positions[rec_index]
                print(f"Adding rec position {i}:", rec_position)
                self.addGlobalVariable(f"rec_x{i}", rec_position[0])
                self.addGlobalVariable(f"rec_y{i}", rec_position[1])
                self.addGlobalVariable(f"rec_z{i}", rec_position[2])
        
        self.addGlobalVariable("num_lig_atoms", 0)
        self.addGlobalVariable("lig_com_x", 0)
        self.addGlobalVariable("lig_com_y", 0)
        self.addGlobalVariable("lig_com_z", 0)
        if self.doing_rec:
            self.addGlobalVariable("num_rec_atoms", 0)
        
        self.addGlobalVariable("rec_com_x", 0)
        self.addGlobalVariable("rec_com_y", 0)
        self.addGlobalVariable("rec_com_z", 0)
        self.addGlobalVariable("com_com_dist_sq", 0)
        
        # Add per-DOF variables for COM-COM calc
        self.addPerDofVariable("is_lig_atom", 0)
        if self.doing_rec:
            self.addPerDofVariable("is_rec_atom", 0)
        return
    
    def prepare_integration_quantities(self):
        # This must be filled out by the sub-class
        raise Exception("The RamdIntegrator object is used as a base class "\
                        "only. It may not be used as an actual integrator.")
    
    def perform_first_step_procedures(self):
        # Perform procedure at first step
        self.beginIfBlock("step_count = 0")
        for i in range(len(self.lig_indices)):
            self.addComputePerDof(
                "is_lig_atom", "is_lig_atom +"\
                +f" (step(_x(x) - (lig_x{i} - eps)))"\
                +f"*(step((lig_x{i} + eps) - _x(x)))"\
                +f"*(step(_y(x) - (lig_y{i} - eps)))"\
                +f"*(step((lig_y{i} + eps) - _y(x)))"\
                +f"*(step(_z(x) - (lig_z{i} - eps)))"\
                +f"*(step((lig_z{i} + eps) - _z(x)))/3")
        
        self.addComputeSum("num_lig_atoms", "is_lig_atom")
        self.addComputeSum("lig_mass", "is_lig_atom * m")
        if self.doing_rec:
            for i in range(len(self.rec_indices)):
                self.addComputePerDof(
                    "is_rec_atom", "is_rec_atom +"\
                    +f" (step(_x(x) - (rec_x{i} - eps)))"\
                    +f"*(step((rec_x{i} + eps) - _x(x)))"\
                    +f"*(step(_y(x) - (rec_y{i} - eps)))"\
                    +f"*(step((rec_y{i} + eps) - _y(x)))"\
                    +f"*(step(_z(x) - (rec_z{i} - eps)))"\
                    +f"*(step((rec_z{i} + eps) - _z(x)))/3")
            
            self.addComputeSum("num_rec_atoms", "is_rec_atom")
            self.addComputeSum("rec_mass", "is_rec_atom * m")
            
        self.endBlock()
        return
        
    def perform_integration_steps(self):
        # This must be filled out by the subclass
        raise Exception("The RamdIntegrator object is used as a base class "\
                        "only. It may not be used as an actual integrator.")
        
    def perform_com_com_calculation(self):
        self.addComputeSum("lig_weighted_x", "is_lig_atom * m * _x(x)")
        self.addComputeSum("lig_weighted_y", "is_lig_atom * m * _y(x)")
        self.addComputeSum("lig_weighted_z", "is_lig_atom * m * _z(x)")
        self.addComputeGlobal("lig_com_x", "lig_weighted_x / lig_mass")
        self.addComputeGlobal("lig_com_y", "lig_weighted_y / lig_mass")
        self.addComputeGlobal("lig_com_z", "lig_weighted_z / lig_mass")
        if self.doing_rec:
            self.addComputeSum("rec_weighted_x", "is_rec_atom * m * _x(x)")
            self.addComputeSum("rec_weighted_y", "is_rec_atom * m * _y(x)")
            self.addComputeSum("rec_weighted_z", "is_rec_atom * m * _z(x)")
            self.addComputeGlobal("rec_com_x", "rec_weighted_x / rec_mass")
            self.addComputeGlobal("rec_com_y", "rec_weighted_y / rec_mass")
            self.addComputeGlobal("rec_com_z", "rec_weighted_z / rec_mass")
        self.addComputeGlobal("com_com_dist_sq", 
                                    " (lig_com_x-rec_com_x)^2+"\
                                    +"(lig_com_y-rec_com_y)^2+"\
                                    +"(lig_com_z-rec_com_z)^2")
        self.addComputeGlobal("step_count", "step_count + 1")
        return
    
    def updateStartingPositions(self, new_positions):
        """
        Run this command after a minimization to update the starting positions
        so that the correct atoms are selected for the COM-COM calculation.
        """
        for i, lig_index in enumerate(self.lig_indices):
            lig_position = new_positions[lig_index]
            self.setGlobalVariableByName(f"lig_x{i}", lig_position[0])
            self.setGlobalVariableByName(f"lig_y{i}", lig_position[1])
            self.setGlobalVariableByName(f"lig_z{i}", lig_position[2])
            
        if self.rec_indices is not None:
            for i, rec_index in enumerate(self.rec_indices):
                rec_position = new_positions[rec_index]
                self.setGlobalVariableByName(f"rec_x{i}", rec_position[0])
                self.setGlobalVariableByName(f"rec_y{i}", rec_position[1])
                self.setGlobalVariableByName(f"rec_z{i}", rec_position[2])
            
        return
    
class RamdLangevinIntegrator(RamdIntegrator):
    def __init__(self, temperature, friction_coeff, dt, lig_indices, 
                 rec_indices, starting_positions):
        # Initialize the superclass integrator
        self.temperature = temperature
        self.friction_coeff = friction_coeff
        super(RamdLangevinIntegrator, self).__init__(
            dt, lig_indices, rec_indices, starting_positions)
        return
        
    def prepare_integration_quantities(self):
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        thermal_energy = kB * self.temperature
        
        # Add global variables for Langevin integration
        self.addGlobalVariable("thermal_energy", thermal_energy)
        self.addGlobalVariable("friction_coeff", self.friction_coeff)
        self.addGlobalVariable("vscale", 0.0)
        self.addGlobalVariable("fscale", 0.0)
        self.addGlobalVariable("noisescale", 0.0)
        
        # Add per-DOF variables for Langevin integration
        self.addPerDofVariable("newx", 0)
        return
        
    def perform_integration_steps(self):
        # perform Langevin self steps
        self.addComputeGlobal("vscale", "exp(-dt*friction_coeff)")
        self.addComputeGlobal("fscale", "(1-vscale)/friction_coeff")
        self.addComputeGlobal("noisescale", 
                                    "sqrt(thermal_energy*(1-vscale*vscale))")
        
        self.addComputePerDof("newx", "x")
        self.addComputePerDof("v", 
                              "vscale*v + fscale*f/m "\
                              "+ noisescale*gaussian/sqrt(m)")
        self.addComputePerDof("x", "x+dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-newx)/dt")
        return
    
    def updateStartingPositions(self, new_positions):
        """
        Run this command after a minimization to update the starting positions
        so that the correct atoms are selected for the COM-COM calculation.
        """
        super(RamdLangevinIntegrator, self).updateStartingPositions(
            new_positions)
        return
    
    def getFriction(self):
        return self.friction
    
    def setFriction(self, friction_coeff):
        self.friction_coeff = friction_coeff
        self.setGlobalVariableByName("friction_coeff", friction_coeff)
        return
    
    def getTemperature(self):
        return self.temperature
    
    def setTemperature(self, temperature):
        self.temperature = temperature
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        thermal_energy = kB * temperature
        self.setGlobalVariableByName("thermal_energy", thermal_energy)
        return