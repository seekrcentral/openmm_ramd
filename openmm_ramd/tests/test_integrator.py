"""
Test integrator.py
"""

import os
import time

import numpy as np
import pytest

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
import parmed
import matplotlib.pyplot as plt

import openmm_ramd.openmm_ramd as openmm_ramd
import openmm_ramd.integrator as ramd_integrator

TEST_DIRECTORY = os.path.dirname(__file__)
ROOT_DIRECTORY = os.path.join(TEST_DIRECTORY, "..")

harmonic_energy = "0.5*k*(x1^2 + y1^2 + z1^2)"
slanted_energy = "k*(x1 + y1 + z1)"

temperature = 300 * unit.kelvin
diffusion_coeff = 0.01 * unit.nanometers**2 / unit.picoseconds
timestep = 0.002 * unit.picoseconds
starting_positions = [[0.0, 0.0, 0.0]]

def prepare_force(potential_energy_expression, 
                  force_const=5.0*unit.kilocalories_per_mole, num_particles=1):
    force = openmm.CustomCentroidBondForce(
        num_particles, potential_energy_expression)
    force.addGlobalParameter("k", force_const)
    return force

class Toy_simulation():
    def __init__(self, temperature, diffusion_coeff, timestep, 
                 starting_positions):
        self.temperature = temperature
        self.timestep = timestep
        self.starting_positions = starting_positions
        self.particle_molar_mass = 10.0*unit.amu
        particle_mass = self.particle_molar_mass / unit.AVOGADRO_CONSTANT_NA
        self.particle_mass = particle_mass.in_units_of(unit.kilograms)
        kT = temperature * unit.BOLTZMANN_CONSTANT_kB
        beta = 1.0/(kT)
        self.beta = beta.in_units_of(unit.picoseconds**2 \
                                / (unit.nanometer**2 * unit.kilograms))
        self.friction_coeff = 1.0 / (self.beta*diffusion_coeff*self.particle_mass)
        return
    
    def prepare_simulation(self, force, num_particles=1, 
                           use_openmm_Brownian=True):
        "Simulate a single particle in the double well"
        system = openmm.System()
        topology = openmm_app.Topology()
        groups_list = []
        lig_indices = list(range(num_particles))
        for i in range(num_particles):
            system.addParticle(self.particle_molar_mass)
            atom_name = "X{}".format(i)
            if not atom_name in openmm_app.element.Element._elements_by_symbol:
                element = openmm_app.element.Element(0, atom_name, atom_name, 
                                                     self.particle_molar_mass)
            else:
                element = openmm_app.element.Element._elements_by_symbol[atom_name]
            chain = topology.addChain()
            residue = topology.addResidue("UNK", chain)
            topology.addAtom(atom_name, element, residue)
            groups_list.append(force.addGroup([i]))
    
        force.addBond(groups_list, [])
        system.addForce(force)
        if use_openmm_Brownian:
            integrator = openmm.BrownianIntegrator(
                self.temperature, self.friction_coeff, self.timestep)
            print("Using Native Integrator")
        else:
            integrator = ramd_integrator.RamdBrownianIntegrator(
                self.temperature, self.friction_coeff, self.timestep, lig_indices, 
                None, self.starting_positions)
            print("Using RAMD integrator")
        
        #simulation = openmm_ramd.RAMDSimulation(
        #    topology, system, integrator, self.ramd_force_magnitude, lig_indices, 
        #    None, platform=None, ramd_steps=ramd_steps, 
        #    force_out_freq=self.steps_per_ramd_update, r_min_ramd=self.rMinRamd, 
        #    max_dist=self.max_dist, position_print_freq=self.position_print_freq)
        simulation = openmm_app.Simulation(topology, system, integrator)
        
        # simulation.reporters.append(openmm_app.PDBReporter(
        #    trajectory_filename, traj_interval))
        # simulation.reporters.append(openmm_app.StateDataReporter(
        #    sys.stdout, traj_interval, step=True, potentialEnergy=True, 
        #    temperature=True))
        self.simulation = simulation
        self.integrator = integrator
        return
    
    def run_simulation(self, num_sims, num_steps):
        end_points = []
        end_energies = []
        for i in range(num_sims):
            self.simulation.context.setPositions(self.starting_positions)
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
            self.simulation.context.reinitialize(preserveState=True)
            self.simulation.step(num_steps)
            state = self.simulation.context.getState(getPositions=True, 
                                                     getEnergy=True)
            positions = state.getPositions()
            end_points.append(positions[0][2].value_in_unit(unit.nanometers))
            pot_energy = state.getPotentialEnergy().value_in_unit(
                unit.kilocalories_per_mole)
            end_energies.append(pot_energy)
            
        return end_points, end_energies
    
def generate_openmm_brownian_points(num_sims, num_steps, 
                                    potential_energy_expression):
    energy_landscape = prepare_force(potential_energy_expression)
    toy = Toy_simulation(
        temperature, diffusion_coeff, timestep, starting_positions)
    toy.prepare_simulation(energy_landscape, use_openmm_Brownian=True)
    return toy.run_simulation(num_sims, num_steps)
    
def generate_ramd_brownian_points(num_sims, num_steps, 
                                  potential_energy_expression):
    energy_landscape = prepare_force(potential_energy_expression)
    toy = Toy_simulation(
        temperature, diffusion_coeff, timestep, starting_positions)
    toy.prepare_simulation(energy_landscape, use_openmm_Brownian=False)
    return toy.run_simulation(num_sims, num_steps)
    
def test_Brownian_harmonic_distributions():
    num_sims = 100
    num_steps = 1000
    native_points, native_energies = generate_openmm_brownian_points(
        num_sims, num_steps, harmonic_energy)
    ramd_points, ramd_energies = generate_ramd_brownian_points(
        num_sims, num_steps, harmonic_energy)
    
    native_points_mean = np.mean(native_points)
    native_points_std = np.std(native_points)
    ramd_points_mean = np.mean(ramd_points)
    ramd_points_std = np.std(ramd_points)
    
    #print("native_points_mean:", native_points_mean)
    #print("ramd_points_mean:", ramd_points_mean)
    #print("native_points_std:", native_points_std)
    #print("ramd_points_std:", ramd_points_std)
    
    native_energy_mean = np.mean(native_energies)
    native_energy_std = np.std(native_energies)
    ramd_energy_mean = np.mean(ramd_energies)
    ramd_energy_std = np.std(ramd_energies)
    
    #print("native_energy_mean:", native_energy_mean)
    #print("ramd_energy_mean:", ramd_energy_mean)
    #print("native_energy_std:", native_energy_std)
    #print("ramd_energy_std:", ramd_energy_std)
    
    #fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # We can set the number of bins with the *bins* keyword argument.
    #axs[0].hist(native_energies, bins=20)
    #axs[1].hist(ramd_energies, bins=20)
    #plt.show()
    
    assert np.isclose(native_points_mean, ramd_points_mean, atol=0.1)
    assert np.isclose(native_points_std, ramd_points_std, rtol=0.25)
    
    assert np.isclose(native_energy_mean, ramd_energy_mean, rtol=0.25)
    assert np.isclose(native_energy_std, ramd_energy_std, rtol=0.5)
    return
    
def test_Brownian_linear_distributions():
    num_sims = 100
    num_steps = 1000
    native_points, native_energies = generate_openmm_brownian_points(
        num_sims, num_steps, slanted_energy)
    ramd_points, ramd_energies = generate_ramd_brownian_points(
        num_sims, num_steps, slanted_energy)
    
    native_points_mean = np.mean(native_points)
    native_points_std = np.std(native_points)
    ramd_points_mean = np.mean(ramd_points)
    ramd_points_std = np.std(ramd_points)
    
    #print("native_points_mean:", native_points_mean)
    #print("ramd_points_mean:", ramd_points_mean)
    #print("native_points_std:", native_points_std)
    #print("ramd_points_std:", ramd_points_std)
    
    native_energy_mean = np.mean(native_energies)
    native_energy_std = np.std(native_energies)
    ramd_energy_mean = np.mean(ramd_energies)
    ramd_energy_std = np.std(ramd_energies)
    
    #print("native_energy_mean:", native_energy_mean)
    #print("ramd_energy_mean:", ramd_energy_mean)
    #print("native_energy_std:", native_energy_std)
    #print("ramd_energy_std:", ramd_energy_std)
    
    #fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # We can set the number of bins with the *bins* keyword argument.
    #axs[0].hist(native_energies, bins=20)
    #axs[1].hist(ramd_energies, bins=20)
    #plt.show()
    
    assert np.isclose(native_energy_mean, ramd_energy_mean, rtol=0.25)
    assert np.isclose(native_energy_std, ramd_energy_std, rtol=0.25)
    
    assert np.isclose(native_points_mean, ramd_points_mean, rtol=0.25)
    assert np.isclose(native_points_std, ramd_points_std, rtol=0.25)
    return

def test_NoseHoover_matches_openmm_reference():
    if not hasattr(openmm, "NoseHooverIntegrator"):
        pytest.skip("OpenMM NoseHooverIntegrator is not available")

    system = openmm.System()
    system.addParticle(12.0*unit.dalton)
    system.addParticle(16.0*unit.dalton)
    system.addConstraint(0, 1, 1.0*unit.nanometer)

    positions = [
        openmm.Vec3(0.0, 0.0, 0.0)*unit.nanometer,
        openmm.Vec3(1.0, 0.0, 0.0)*unit.nanometer,
    ]
    velocities = [
        openmm.Vec3(0.10, 0.20, 0.00)*unit.nanometer/unit.picosecond,
        openmm.Vec3(0.10, -0.10, 0.05)*unit.nanometer/unit.picosecond,
    ]
    temperature = 300.0*unit.kelvin
    collision_frequency = 1.0/unit.picosecond
    step_size = 0.001*unit.picoseconds
    platform = openmm.Platform.getPlatformByName("Reference")

    native = openmm.NoseHooverIntegrator(
        temperature, collision_frequency, step_size)
    custom = ramd_integrator.RamdNoseHooverIntegrator(
        temperature, collision_frequency, step_size, [0], [1], positions,
        system=system)
    assert custom.getNumDegreesOfFreedom() == 5

    native_context = openmm.Context(system, native, platform)
    native_context.setPositions(positions)
    native_context.setVelocities(velocities)

    custom_context = openmm.Context(system, custom, platform)
    custom_context.setPositions(positions)
    custom_context.setVelocities(velocities)

    for step in range(25):
        native.step(1)
        custom.step(1)

        native_state = native_context.getState(
            getPositions=True, getVelocities=True)
        custom_state = custom_context.getState(
            getPositions=True, getVelocities=True)

        native_positions = native_state.getPositions(
            asNumpy=True).value_in_unit(unit.nanometer)
        custom_positions = custom_state.getPositions(
            asNumpy=True).value_in_unit(unit.nanometer)
        native_velocities = native_state.getVelocities(
            asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        custom_velocities = custom_state.getVelocities(
            asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        native_heat_bath_energy = native.computeHeatBathEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        custom_heat_bath_energy = custom.computeHeatBathEnergy().value_in_unit(
            unit.kilojoule_per_mole)

        assert np.allclose(custom_positions, native_positions, atol=1e-12), step
        assert np.allclose(custom_velocities, native_velocities, atol=1e-12), step
        assert np.isclose(
            custom_heat_bath_energy, native_heat_bath_energy, atol=1e-12), step
    return
