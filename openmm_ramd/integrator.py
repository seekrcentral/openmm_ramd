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
        self.addGlobalVariable("pos_margin", 0.001 * unit.nanometers)
        for i, lig_index in enumerate(self.lig_indices):
            lig_position = self.starting_positions[lig_index]
            self.addGlobalVariable(f"lig_x{i}", lig_position[0])
            self.addGlobalVariable(f"lig_y{i}", lig_position[1])
            self.addGlobalVariable(f"lig_z{i}", lig_position[2])
        
        if self.doing_rec:
            for i, rec_index in enumerate(self.rec_indices):
                rec_position = self.starting_positions[rec_index]
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
                +f" (step(_x(x) - (lig_x{i} - pos_margin)))"\
                +f"*(step((lig_x{i} + pos_margin) - _x(x)))"\
                +f"*(step(_y(x) - (lig_y{i} - pos_margin)))"\
                +f"*(step((lig_y{i} + pos_margin) - _y(x)))"\
                +f"*(step(_z(x) - (lig_z{i} - pos_margin)))"\
                +f"*(step((lig_z{i} + pos_margin) - _z(x)))/3")
        
        self.addComputeSum("num_lig_atoms", "is_lig_atom")
        self.addComputeSum("lig_mass", "is_lig_atom * m")
        if self.doing_rec:
            for i in range(len(self.rec_indices)):
                self.addComputePerDof(
                    "is_rec_atom", "is_rec_atom +"\
                    +f" (step(_x(x) - (rec_x{i} - pos_margin)))"\
                    +f"*(step((rec_x{i} + pos_margin) - _x(x)))"\
                    +f"*(step(_y(x) - (rec_y{i} - pos_margin)))"\
                    +f"*(step((rec_y{i} + pos_margin) - _y(x)))"\
                    +f"*(step(_z(x) - (rec_z{i} - pos_margin)))"\
                    +f"*(step((rec_z{i} + pos_margin) - _z(x)))/3")
            
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
        self.addPerDofVariable("oldx", 0)
        return
        
    def perform_integration_steps(self):
        # perform Langevin steps
        self.addUpdateContextState()
        self.addComputeGlobal("vscale", "exp(-dt*friction_coeff)")
        self.addComputeGlobal("fscale", "(1-vscale)/friction_coeff")
        self.addComputeGlobal("noisescale", 
                                    "sqrt(thermal_energy*(1-vscale*vscale))")
        
        self.addComputePerDof("oldx", "x")
        self.addComputePerDof("v", 
                              "vscale*v + fscale*f/m "\
                              "+ noisescale*gaussian/sqrt(m)")
        self.addComputePerDof("x", "x+dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-oldx)/dt")
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
        return self.friction_coeff
    
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
        
class RamdLangevinMiddleIntegrator(RamdIntegrator):
    def __init__(self, temperature, friction_coeff, dt, lig_indices, 
                 rec_indices, starting_positions):
        # Initialize the superclass integrator
        self.temperature = temperature
        self.friction_coeff = friction_coeff
        super(RamdLangevinMiddleIntegrator, self).__init__(
            dt, lig_indices, rec_indices, starting_positions)
        return
        
    def prepare_integration_quantities(self):
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        thermal_energy = kB * self.temperature
        
        # Add global variables for Langevin Middle integration
        friction_unitless = self.friction_coeff.value_in_unit(unit.picoseconds**-1)
        timestep_unitless = self.getStepSize().value_in_unit(unit.picoseconds)
        self.addGlobalVariable("a", np.exp(-friction_unitless*timestep_unitless))
        self.addGlobalVariable("b", np.sqrt(1-np.exp(-2*friction_unitless*timestep_unitless)))
        self.addGlobalVariable("thermal_energy", thermal_energy)
        #self.addGlobalVariable("thermal_energy", thermal_energy)
        #self.addGlobalVariable("friction_coeff", self.friction_coeff)
        #self.addGlobalVariable("vscale", 0.0)
        #self.addGlobalVariable("fscale", 0.0)
        #self.addGlobalVariable("noisescale", 0.0)
        
        # Add per-DOF variables for Langevin integration
        self.addPerDofVariable("x1", 0)
        return
        
    def perform_integration_steps(self):
        # perform Langevin steps
        self.addUpdateContextState()
        self.addComputePerDof("v", "v + dt*f/m")
        self.addConstrainVelocities()
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + b*sqrt(thermal_energy/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")
        return
    
    def updateStartingPositions(self, new_positions):
        """
        Run this command after a minimization to update the starting positions
        so that the correct atoms are selected for the COM-COM calculation.
        """
        super(RamdLangevinMiddleIntegrator, self).updateStartingPositions(
            new_positions)
        return
    
    def getFriction(self):
        return self.friction_coeff
    
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

class RamdNoseHooverIntegrator(RamdIntegrator):
    """
    RAMD CustomIntegrator version of OpenMM's NoseHooverIntegrator.

    This implements the same LF-middle splitting and Nose-Hoover chain
    propagation used by OpenMM's C++ implementation for a full-system
    thermostat.  Pass system, or pass num_dofs directly, to match the native
    C++ degree-of-freedom count for constrained systems and systems with center
    of mass motion removal.  If neither is provided, the integrator counts
    massive degrees of freedom on the first step.
    """
    def __init__(self, temperature, collision_frequency, dt, lig_indices,
                 rec_indices, starting_positions, chain_length=3, num_mts=3,
                 num_yoshida_suzuki=7, num_dofs=None, system=None):
        # Initialize the superclass integrator
        self.temperature = temperature
        self.collision_frequency = collision_frequency
        self.chain_length = int(chain_length)
        self.num_mts = int(num_mts)
        self.num_yoshida_suzuki = int(num_yoshida_suzuki)
        self.num_dofs = num_dofs
        if self.num_dofs is None and system is not None:
            self.num_dofs = self._compute_full_system_num_dofs(system)
        self._validate_parameters()
        super(RamdNoseHooverIntegrator, self).__init__(
            dt, lig_indices, rec_indices, starting_positions)
        return

    @staticmethod
    def _value_in_unit(value, unit_):
        try:
            return value.value_in_unit(unit_)
        except AttributeError:
            return value

    def _compute_full_system_num_dofs(self, system):
        massive_particles = set()
        for particle in range(system.getNumParticles()):
            mass = system.getParticleMass(particle)
            if self._value_in_unit(mass, unit.dalton) > 0:
                massive_particles.add(particle)

        num_dofs = 3*len(massive_particles)
        for constraint_index in range(system.getNumConstraints()):
            particle1, particle2, distance = system.getConstraintParameters(
                constraint_index)
            if particle1 in massive_particles and particle2 in massive_particles:
                num_dofs -= 1

        for force_index in range(system.getNumForces()):
            if isinstance(system.getForce(force_index), openmm.CMMotionRemover):
                num_dofs -= 3
        return num_dofs

    def _validate_parameters(self):
        temperature_value = self._value_in_unit(self.temperature, unit.kelvin)
        frequency_value = self._value_in_unit(
            self.collision_frequency, unit.picoseconds**-1)
        if temperature_value < 0:
            raise ValueError("temperature cannot be negative")
        if frequency_value <= 0:
            raise ValueError("collision_frequency must be positive")
        if self.chain_length <= 0:
            raise ValueError("chain_length must be positive")
        if self.num_mts <= 0:
            raise ValueError("num_mts must be positive")
        if self.num_yoshida_suzuki not in (1, 3, 5, 7):
            raise ValueError("num_yoshida_suzuki must be 1, 3, 5, or 7")
        if self.num_dofs is not None and self.num_dofs <= 0:
            raise ValueError("num_dofs must be positive")
        return

    def _get_yoshida_suzuki_weights(self):
        if self.num_yoshida_suzuki == 1:
            return [1.0]
        if self.num_yoshida_suzuki == 3:
            return [0.828981543588751, -0.657963087177502,
                    0.828981543588751]
        if self.num_yoshida_suzuki == 5:
            return [0.2967324292201065, 0.2967324292201065,
                    -0.186929716880426, 0.2967324292201065,
                    0.2967324292201065]
        return [0.784513610477560, 0.235573213359357,
                -1.17767998417887, 1.31518632068391,
                -1.17767998417887, 0.235573213359357,
                0.784513610477560]

    def prepare_integration_quantities(self):
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        thermal_energy = kB * self.temperature
        num_dofs = 0 if self.num_dofs is None else self.num_dofs

        # Add global variables for Nose-Hoover chain integration
        self.addGlobalVariable("nh_thermal_energy", thermal_energy)
        self.addGlobalVariable("nh_collision_frequency",
                               self.collision_frequency)
        self.addGlobalVariable("nh_num_dofs", num_dofs)
        self.addGlobalVariable("nh_kinetic_energy", 0)
        self.addGlobalVariable("nh_twice_kinetic_energy", 0)
        self.addGlobalVariable("nh_scale", 1)
        self.addGlobalVariable("nh_wdt", 0)
        self.addGlobalVariable("nh_aa", 0)
        self.addGlobalVariable("nh_heat_bath_energy", 0)
        for bead in range(self.chain_length):
            self.addGlobalVariable(f"nh_chain_pos_{bead}", 0)
            self.addGlobalVariable(f"nh_chain_vel_{bead}", 0)
            self.addGlobalVariable(f"nh_chain_mass_{bead}", 0)
            self.addGlobalVariable(f"nh_chain_force_{bead}", 0)

        # Add per-DOF variables for LF-middle integration
        self.addPerDofVariable("x1", 0)
        self.setKineticEnergyExpression("0.5*m*v*v")
        return

    def _add_update_chain_masses(self):
        frequency_sq = "nh_collision_frequency*nh_collision_frequency"
        self.addComputeGlobal(
            "nh_chain_mass_0",
            f"nh_num_dofs*nh_thermal_energy/({frequency_sq})")
        for bead in range(1, self.chain_length):
            self.addComputeGlobal(
                f"nh_chain_mass_{bead}",
                f"nh_thermal_energy/({frequency_sq})")
        return

    def _add_initial_chain_forces(self):
        self.addComputeGlobal(
            "nh_chain_force_0",
            "(nh_twice_kinetic_energy - nh_num_dofs*nh_thermal_energy)"
            "/nh_chain_mass_0")
        for bead in range(self.chain_length - 1):
            self.addComputeGlobal(
                f"nh_chain_force_{bead+1}",
                f"(nh_chain_mass_{bead}*nh_chain_vel_{bead}"
                f"*nh_chain_vel_{bead} - nh_thermal_energy)"
                f"/nh_chain_mass_{bead+1}")
        return

    def _add_propagate_chain(self):
        last_bead = self.chain_length - 1
        for mts in range(self.num_mts):
            for weight in self._get_yoshida_suzuki_weights():
                self.addComputeGlobal(
                    "nh_wdt", f"{weight:.16g}*dt/{self.num_mts}")
                self.addComputeGlobal(
                    f"nh_chain_vel_{last_bead}",
                    f"nh_chain_vel_{last_bead}"
                    f" + 0.5*nh_wdt*nh_chain_force_{last_bead}")
                for bead in range(self.chain_length - 2, -1, -1):
                    self.addComputeGlobal(
                        "nh_aa",
                        f"exp(-0.25*nh_wdt*nh_chain_vel_{bead+1})")
                    self.addComputeGlobal(
                        f"nh_chain_vel_{bead}",
                        f"nh_aa*(nh_chain_vel_{bead}*nh_aa"
                        f" + 0.5*nh_wdt*nh_chain_force_{bead})")
                self.addComputeGlobal(
                    "nh_scale", "nh_scale*exp(-nh_wdt*nh_chain_vel_0)")
                for bead in range(self.chain_length):
                    self.addComputeGlobal(
                        f"nh_chain_pos_{bead}",
                        f"nh_chain_pos_{bead}"
                        f" + nh_chain_vel_{bead}*nh_wdt")
                self.addComputeGlobal(
                    "nh_chain_force_0",
                    "(nh_scale*nh_scale*nh_twice_kinetic_energy"
                    " - nh_num_dofs*nh_thermal_energy)/nh_chain_mass_0")
                for bead in range(self.chain_length - 1):
                    self.addComputeGlobal(
                        "nh_aa",
                        f"exp(-0.25*nh_wdt*nh_chain_vel_{bead+1})")
                    self.addComputeGlobal(
                        f"nh_chain_vel_{bead}",
                        f"nh_aa*(nh_aa*nh_chain_vel_{bead}"
                        f" + 0.5*nh_wdt*nh_chain_force_{bead})")
                    self.addComputeGlobal(
                        f"nh_chain_force_{bead+1}",
                        f"(nh_chain_mass_{bead}*nh_chain_vel_{bead}"
                        f"*nh_chain_vel_{bead} - nh_thermal_energy)"
                        f"/nh_chain_mass_{bead+1}")
                self.addComputeGlobal(
                    f"nh_chain_vel_{last_bead}",
                    f"nh_chain_vel_{last_bead}"
                    f" + 0.5*nh_wdt*nh_chain_force_{last_bead}")
        return

    def _add_compute_heat_bath_energy(self):
        energy_terms = []
        for bead in range(self.chain_length):
            prefac = "nh_num_dofs" if bead == 0 else "1"
            energy_terms.append(
                f"0.5*nh_chain_mass_{bead}*nh_chain_vel_{bead}"
                f"*nh_chain_vel_{bead}"
                f" + {prefac}*nh_thermal_energy*nh_chain_pos_{bead}")
        self.addComputeGlobal("nh_heat_bath_energy", " + ".join(energy_terms))
        return

    def perform_integration_steps(self):
        # Perform Nose-Hoover LF-middle steps
        self.addUpdateContextState()
        if self.num_dofs is None:
            self.beginIfBlock("step_count = 0")
            self.addComputeSum("nh_num_dofs", "1")
            self.endBlock()

        self._add_update_chain_masses()
        self.addComputePerDof("v", "v + dt*f/m")
        self.addConstrainVelocities()
        self.addComputePerDof("x", "x + 0.5*dt*v")

        self.addComputeSum("nh_kinetic_energy", "0.5*m*v*v")
        self.addComputeGlobal("nh_twice_kinetic_energy",
                              "2*nh_kinetic_energy")
        self.addComputeGlobal("nh_scale", "1")
        self.beginIfBlock("nh_kinetic_energy >= 1e-8")
        self._add_initial_chain_forces()
        self._add_propagate_chain()
        self.endBlock()
        self.addComputePerDof("v", "nh_scale*v")

        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")
        self._add_compute_heat_bath_energy()
        return

    def updateStartingPositions(self, new_positions):
        """
        Run this command after a minimization to update the starting positions
        so that the correct atoms are selected for the COM-COM calculation.
        """
        super(RamdNoseHooverIntegrator, self).updateStartingPositions(
            new_positions)
        return

    def getTemperature(self):
        return self.temperature

    def setTemperature(self, temperature):
        self.temperature = temperature
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        thermal_energy = kB * temperature
        self.setGlobalVariableByName("nh_thermal_energy", thermal_energy)
        return

    def getCollisionFrequency(self):
        return self.collision_frequency

    def setCollisionFrequency(self, collision_frequency):
        self.collision_frequency = collision_frequency
        self.setGlobalVariableByName(
            "nh_collision_frequency", collision_frequency)
        return

    def getFriction(self):
        return self.getCollisionFrequency()

    def setFriction(self, friction_coeff):
        self.setCollisionFrequency(friction_coeff)
        return

    def getNumDegreesOfFreedom(self):
        return self.getGlobalVariableByName("nh_num_dofs")

    def setNumDegreesOfFreedom(self, num_dofs):
        if num_dofs <= 0:
            raise ValueError("num_dofs must be positive")
        self.num_dofs = num_dofs
        self.setGlobalVariableByName("nh_num_dofs", num_dofs)
        return

    def computeHeatBathEnergy(self):
        return (self.getGlobalVariableByName("nh_heat_bath_energy")
                * unit.kilojoule_per_mole)

    def getChainState(self):
        positions = []
        velocities = []
        for bead in range(self.chain_length):
            positions.append(
                self.getGlobalVariableByName(f"nh_chain_pos_{bead}"))
            velocities.append(
                self.getGlobalVariableByName(f"nh_chain_vel_{bead}"))
        return positions, velocities

class RamdBrownianIntegrator(RamdIntegrator):
    def __init__(self, temperature, friction_coeff, dt, lig_indices, 
                 rec_indices, starting_positions):
        # Initialize the superclass integrator
        self.temperature = temperature
        self.friction_coeff = friction_coeff
        super(RamdBrownianIntegrator, self).__init__(
            dt, lig_indices, rec_indices, starting_positions)
        return
        
    def prepare_integration_quantities(self):
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        thermal_energy = kB * self.temperature
        
        # Add global variables for Brownian integration
        self.addGlobalVariable("thermal_energy", thermal_energy)
        self.addGlobalVariable("friction_coeff", self.friction_coeff)
        self.addGlobalVariable("fscale", 0.0)
        self.addGlobalVariable("noiseAmplitude", 0.0)
        
        # Add per-DOF variables for Brownian integration
        self.addPerDofVariable("oldx", 0)
        return
        
    def perform_integration_steps(self):
        # perform Brownian steps
        self.addUpdateContextState()
        self.addComputeGlobal("fscale", "dt/friction_coeff")
        self.addComputeGlobal("noiseAmplitude", 
                                    "sqrt(2*thermal_energy*dt/friction_coeff)")
        self.addComputePerDof("oldx", "x")
        self.addComputePerDof("x", "x + fscale*f/m + noiseAmplitude*gaussian/sqrt(m)")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-oldx)/dt")
        return
    
    def updateStartingPositions(self, new_positions):
        """
        Run this command after a minimization to update the starting positions
        so that the correct atoms are selected for the COM-COM calculation.
        """
        super(RamdBrownianIntegrator, self).updateStartingPositions(
            new_positions)
        return
    
    def getFriction(self):
        return self.friction_coeff
    
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

class RamdVerletIntegrator(RamdIntegrator):
    def __init__(self, dt, lig_indices, 
                 rec_indices, starting_positions):
        # Initialize the superclass integrator
        super(RamdVerletIntegrator, self).__init__(
            dt, lig_indices, rec_indices, starting_positions)
        return
        
    def prepare_integration_quantities(self):
        # Add per-DOF variables for Verlet integration
        self.addPerDofVariable("old_x", 0)
        return
        
    def perform_integration_steps(self):
        # perform Verlet steps
        self.addUpdateContextState()
        self.addComputePerDof("old_x", "x")
        self.addComputePerDof("v", "v+dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-old_x)/dt")
        return
    
    def updateStartingPositions(self, new_positions):
        """
        Run this command after a minimization to update the starting positions
        so that the correct atoms are selected for the COM-COM calculation.
        """
        super(RamdVerletIntegrator, self).updateStartingPositions(
            new_positions)
        return
    
