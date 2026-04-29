"""
A series of helper function for preparing and running RAMD simulations.

Includes:
 - Functions for creating systems and simulation objects from different 
   input file types.
"""

import sys
import os
import time

import numpy as np
import mdtraj
import openmm
import openmm.app as openmm_app
import openmm.unit as unit

import openmm_ramd.base as openmm_ramd_base
import openmm_ramd.integrator as ramd_integrator
import openmm_ramd.openmm_ramd as ramd_simulation
import openmm_ramd.analyze.base as analyze_base

ALLOWED_SIM_PARAMETER_KEYS = {
    "temperature",
    "time_step",
    "nonbonded_cutoff",
    "pressure",
    "friction_coeff",
    "hydrogen_mass",
    "random_seed",
    "integrator_type",
    "restraint_force_constant",
    "restraint_indices",
    "steps_per_energy_update",
    "steps_per_trajectory_update",
}

ALLOWED_RAMD_PARAMETER_KEYS = {
    "ligand_indices",
    "receptor_indices",
    "starting_positions",
    "steps_per_RAMD_update",
    "position_print_freq",
    "ramd_force_magnitude",
    "RAMD_cutoff_distance",
    "RAMD_max_distance",
    "max_travel",
    "max_num_steps",
    "ramp_start_ns",
    "ramp_force_increase_per_ns",
    "num_starting_structures",
    "num_velocity_restarts",
    "steps_between_structures",
}

def validate_sim_parameters(sim_parameters: dict):
    for sim_param_key in sim_parameters.keys():
        assert sim_param_key in ALLOWED_SIM_PARAMETER_KEYS, \
            f"sim_parameters should only have keys {ALLOWED_SIM_PARAMETER_KEYS}"

def validate_ramd_parameters(ramd_parameters: dict):
    for ramd_param_key in ramd_parameters.keys():
        assert ramd_param_key in ALLOWED_RAMD_PARAMETER_KEYS, \
            f"ramd_parameters should only have keys {ALLOWED_RAMD_PARAMETER_KEYS}"

def should_run(log_filename):
    if not os.path.exists(log_filename):
        # The file doesn't exist yet - it needs to be run
        return True
    
    elif analyze_base.escaped_successfully(log_filename):
        # This trajectory already escaped - no need to run again.
        return False
    
    elif analyze_base.ran_out_of_time(log_filename):
        # This trajectory already ran and reached maximum time
        # - Don't run again
        return False
    
    else:
        # Something else must have gone wrong with this trajectory
        # attempt to run again
        return True

def make_system_from_amber_inputs(
        prmtop_filename: str,
        sim_parameters: dict,
        ) -> openmm.System:
    """
    Make an OpenMM System object from Amber prmtop and inpcrd files.
    """
    prmtop = openmm_app.AmberPrmtopFile(prmtop_filename)
    temperature = sim_parameters["temperature"] * unit.kelvin
    nonbonded_cutoff = sim_parameters.get("nonbonded_cutoff", 0.8) * unit.nanometers
    pressure = sim_parameters.get("pressure", None)
    if pressure is not None:
        pressure = pressure * unit.bar
    #time_step = parameters["time_step"] * unit.picoseconds
    hydrogen_mass = sim_parameters.get("hydrogen_mass", None) * unit.daltons
    system = prmtop.createSystem(
        nonbondedMethod=openmm_app.PME, nonbondedCutoff=nonbonded_cutoff,
        constraints=openmm_app.HBonds, hydrogenMass=hydrogen_mass,)
    if pressure:
        barostat = openmm.MonteCarloBarostat(pressure, temperature, 25)
        system.addForce(barostat)
    return prmtop, system

def make_system_from_openmm_system_xml(
        system_xml_filename: str,
        ) -> openmm.System:
    """
    Make an OpenMM System object from an OpenMM System XML file.
    """
    with open(system_xml_filename) as f:
        system = openmm.XmlSerializer.deserialize(f.read())
    return system

# TODO: charmm, gromacs, etc. input file types

def choose_topology_system_from_params(
        ff_parameters: dict, 
        sim_parameters: dict
        ) -> openmm.System:
    if ff_parameters["type"] == "amber":
        # Validate ff_parameters keys and values for amber
        for ff_param_key in ff_parameters.keys():
            assert ff_param_key in {"type", "prmtop_filename",
                                    "inpcrd_filename", "pdb_filename"}, \
                "ff_parameters for amber should only have keys 'type', "\
                "'prmtop_filename', 'inpcrd_filename', and 'pdb_filename'"
        assert "prmtop_filename" in ff_parameters, \
            "ff_parameters for amber should have key 'prmtop_filename'"
        prmtop, system = make_system_from_amber_inputs(
            ff_parameters["prmtop_filename"],
            sim_parameters,
        )
        topology = prmtop.topology
    elif ff_parameters["type"] == "openmm_system_xml":
        # Validate ff_parameters keys and values for openmm_system_xml
        for ff_param_key in ff_parameters.keys():
            assert ff_param_key in {"type", "system_xml_filename",
                                    "pdb_filename"}, \
                "ff_parameters for openmm_system_xml should only have keys "\
                "'type', 'system_xml_filename', and 'pdb_filename'"
        assert "system_xml_filename" in ff_parameters, \
            "ff_parameters for openmm_system_xml should have key "\
            "'system_xml_filename'"
        system = make_system_from_openmm_system_xml(
            ff_parameters["system_xml_filename"],
        )
        pdb = openmm_app.PDBFile(ff_parameters["pdb_filename"])
        topology = pdb.topology
    # elif ff_parameters["type"] == "charmm": # and gromacs, etc.
    else:
        raise ValueError(f"Unsupported ff_parameters type: {ff_parameters['type']}")
    
    return topology, system

def choose_conventional_integrator(
        sim_parameters: dict
        ) -> openmm.Integrator:
    if sim_parameters["integrator_type"] == "langevin":
        temperature = sim_parameters["temperature"] * unit.kelvin
        time_step = sim_parameters["time_step"] * unit.picoseconds
        friction_coeff = sim_parameters.get("friction_coeff", 1.0) / unit.picosecond
        integrator = openmm.LangevinIntegrator(
            temperature, friction_coeff, time_step)
        random_seed = sim_parameters.get("random_seed", 0)
        integrator.setRandomNumberSeed(random_seed)
    elif sim_parameters["integrator_type"] == "verlet":
        time_step = sim_parameters["time_step"] * unit.picoseconds
        integrator = openmm.VerletIntegrator(time_step)
    elif sim_parameters["integrator_type"] == "brownian":
        temperature = sim_parameters["temperature"] * unit.kelvin
        time_step = sim_parameters["time_step"] * unit.picoseconds
        friction_coeff = sim_parameters.get("friction_coeff", 1.0) / unit.picosecond
        integrator = openmm.BrownianIntegrator(
            temperature, friction_coeff, time_step)
        random_seed = sim_parameters.get("random_seed", 0)
        integrator.setRandomNumberSeed(random_seed)
    else:
        raise ValueError(
            f"Unsupported integrator type: {sim_parameters['integrator_type']}")
    return integrator

def choose_ramd_integrator(
        sim_parameters: dict,
        ramd_parameters: dict,
        starting_positions: list[openmm.Vec3],
        ) -> ramd_integrator.RamdIntegrator:
    lig_indices = ramd_parameters["ligand_indices"]
    rec_indices = ramd_parameters["receptor_indices"]
    if sim_parameters["integrator_type"] == "langevin":
        temperature = sim_parameters["temperature"] * unit.kelvin
        time_step = sim_parameters["time_step"] * unit.picoseconds
        friction_coeff = sim_parameters.get(
            "friction_coeff", 1.0) / unit.picosecond
        integrator = ramd_integrator.RamdLangevinIntegrator(
            temperature, friction_coeff, time_step, lig_indices,
            rec_indices, starting_positions)
        random_seed = sim_parameters.get("random_seed", 0)
        if random_seed != 0:
            random_seed += 1
        integrator.setRandomNumberSeed(random_seed)
    elif sim_parameters["integrator_type"] == "verlet":
        time_step = sim_parameters["time_step"] * unit.picoseconds
        integrator = ramd_integrator.RamdVerletIntegrator(
            time_step, lig_indices,
            rec_indices, starting_positions)
    elif sim_parameters["integrator_type"] == "brownian":
        temperature = sim_parameters["temperature"] * unit.kelvin
        time_step = sim_parameters["time_step"] * unit.picoseconds
        friction_coeff = sim_parameters.get(
            "friction_coeff", 1.0) / unit.picosecond
        integrator = ramd_integrator.RamdBrownianIntegrator(
            temperature, friction_coeff, time_step, lig_indices,
            rec_indices, starting_positions)
        random_seed = sim_parameters.get("random_seed", 0)
        if random_seed != 0:
            random_seed += 1
        integrator.setRandomNumberSeed(random_seed)
    else:
        raise ValueError(
            f"Unsupported integrator type: {sim_parameters['integrator_type']}")
    return integrator

def choose_platform_and_properties(
        cuda_device_index: str | None = "0",
        ) -> tuple[openmm.Platform, dict]:
    if cuda_device_index is None:
        platform = openmm.Platform.getPlatformByName('Reference')
        properties = {}
    else:
        platform = openmm.Platform.getPlatformByName('CUDA')
        properties = {"CudaDeviceIndex": cuda_device_index, 
                      "CudaPrecision":   "mixed"}
    return platform, properties

def add_restraints_to_system(
        system: openmm.System,
        ff_parameters: dict,
        sim_parameters: dict,
        ) -> None:
    pdb = openmm_app.PDBFile(ff_parameters["pdb_filename"])
    if sim_parameters.get("restraint_indices", None) is not None:
        assert sim_parameters.get("restraint_force_constant", None) is not None, \
            "If sim_parameters has key 'restraint_indices', it should also have "\
            "key 'restraint_force_constant'"
        openmm_ramd_base.impose_receptor_restraints(
            system, pdb.positions, 
            sim_parameters["restraint_force_constant"], 
            sim_parameters["restraint_indices"])

def make_conventional_simulation(
        topology: openmm.app.Topology,
        system: openmm.System,
        integrator: openmm.Integrator,
        platform: openmm.Platform,
        properties: dict,
        ff_parameters: dict,
        sim_parameters: dict,
        minimize: bool,
        ) -> openmm.app.Simulation:
    simulation = openmm_app.Simulation(
        topology, system, integrator, platform, properties)
    pdb = openmm_app.PDBFile(ff_parameters["pdb_filename"])
    simulation.context.setPositions(pdb.positions)
    box_vectors = (pdb.topology).getPeriodicBoxVectors()
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    if minimize:
        simulation.minimizeEnergy()
    temperature = sim_parameters["temperature"] * unit.kelvin
    random_seed = sim_parameters.get("random_seed", 0)
    simulation.context.setVelocitiesToTemperature(temperature, random_seed)
    steps_per_energy_update = sim_parameters.get("steps_per_energy_update", None)
    if steps_per_energy_update is not None:
        simulation.reporters.append(openmm_app.StateDataReporter(
            sys.stdout, steps_per_energy_update, step=True,
                potentialEnergy=True, temperature=True, volume=True))
    return simulation

def make_ramd_simulation(
        topology: openmm.app.Topology,
        system: openmm.System,
        integrator: ramd_integrator.RamdIntegrator,
        platform: openmm.Platform,
        properties: dict,
        ff_parameters: dict,
        sim_parameters: dict,
        ramd_parameters: dict,
        starting_structure_name: str,
        minimize: bool,
        ramd_log_filename: str,
        trajectory_filename: str | None = None,
        ) -> openmm.app.Simulation:
    steps_per_RAMD_update = ramd_parameters["steps_per_RAMD_update"]
    position_print_freq = ramd_parameters["position_print_freq"]
    rec_indices = ramd_parameters["receptor_indices"]
    lig_indices = ramd_parameters["ligand_indices"]
    ramd_force_magnitude = ramd_parameters["ramd_force_magnitude"] * unit.kilojoules_per_mole / unit.nanometer
    RAMD_cutoff_distance = ramd_parameters["RAMD_cutoff_distance"] * unit.nanometers
    RAMD_max_distance = ramd_parameters["RAMD_max_distance"] * unit.nanometers
    max_travel = ramd_parameters["max_travel"] * unit.nanometers
    ramp_start_ns = ramd_parameters.get("ramp_start_ns", 0.0)
    ramp_force_increase_per_ns = ramd_parameters.get(
        "ramp_force_increase_per_ns", 0.0)
    ramd_steps = min(steps_per_RAMD_update, position_print_freq)
    
    simulation = ramd_simulation.RAMDSimulation(
        topology, system, integrator, ramd_force_magnitude, lig_indices, 
        rec_indices, ramd_steps=ramd_steps, 
        r_min_ramd=RAMD_cutoff_distance, 
        force_out_freq=steps_per_RAMD_update, 
        max_dist=RAMD_max_distance, max_travel=max_travel,
        platform=platform, properties=properties, log_file_name=ramd_log_filename,
        position_print_freq=position_print_freq, ramp_start_ns=ramp_start_ns,
        ramp_force_increase_per_ns=ramp_force_increase_per_ns)
    
    pdb = openmm_app.PDBFile(starting_structure_name)
    simulation.context.setPositions(pdb.positions)
    box_vectors = (pdb.topology).getPeriodicBoxVectors()
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    if minimize:
        simulation.minimizeEnergy()
    temperature = sim_parameters["temperature"] * unit.kelvin
    random_seed = sim_parameters.get("random_seed", 0)
    simulation.context.setVelocitiesToTemperature(temperature, random_seed)
    steps_per_energy_update = sim_parameters.get("steps_per_energy_update", None)
    steps_per_trajectory_update = sim_parameters.get("steps_per_trajectory_update", None)
    if steps_per_energy_update is not None:
        simulation.reporters.append(openmm_app.StateDataReporter(
            sys.stdout, steps_per_energy_update, step=True,
                potentialEnergy=True, temperature=True, volume=True))
    if steps_per_trajectory_update is not None:
        assert trajectory_filename is not None, \
            "trajectory_filename must be assigned if "\
            "'steps_per_trajectory_update' is not None"
        dcd_reporter = openmm_app.DCDReporter(
            trajectory_filename, steps_per_trajectory_update, enforcePeriodicBox=True)
        simulation.reporters.append(dcd_reporter)
    else:
        assert trajectory_filename is None, \
            "trajectory_filename cannot be assigned if "\
            "'steps_per_trajectory_update' is None"
    return simulation

def write_conformations_to_pdb(
        simulation: openmm.app.Simulation,
        num_structures_to_save: int,
        steps_between_structures: int,
        structure_basename: str,
        rec_indices: list[int],
        ) -> list[str]:
    structure_names = []
    for i in range(num_structures_to_save):
        simulation.step(steps_between_structures)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        ref_positions = state.getPositions(asNumpy=True, )
        positions_list = [ref_positions]
        xyz = np.array(positions_list)
        traj = mdtraj.Trajectory(xyz, mdtraj.Topology.from_openmm(simulation.topology))
        # Set mdtraj's periodic box vectors
        box_vectors = state.getPeriodicBoxVectors()
        traj.unitcell_vectors = np.array(
            [[[box_vectors[0][0].value_in_unit(unit.nanometers), 
              box_vectors[0][1].value_in_unit(unit.nanometers), 
              box_vectors[0][2].value_in_unit(unit.nanometers)],
             [box_vectors[1][0].value_in_unit(unit.nanometers), 
              box_vectors[1][1].value_in_unit(unit.nanometers), 
              box_vectors[1][2].value_in_unit(unit.nanometers)],
             [box_vectors[2][0].value_in_unit(unit.nanometers), 
              box_vectors[2][1].value_in_unit(unit.nanometers), 
              box_vectors[2][2].value_in_unit(unit.nanometers)]]], dtype=np.float32)
        atom_set_list = []
        for index in rec_indices:
            atom_set_list.append(set([traj.topology.atom(index)]))
        assert len(atom_set_list) > 0
        # Convert to float32 to avoid dtype mismatch with mdtraj's C extension
        traj.xyz = traj.xyz.astype(np.float32)
        traj.image_molecules(inplace=True, anchor_molecules=atom_set_list)
        structure_fullname = structure_basename.format(i)
        print("saving structure to", structure_fullname)
        traj.save(os.path.join(structure_fullname))
        structure_names.append(structure_fullname)

    return structure_names

def run_ramd_sim(
        simulation: ramd_simulation.RAMDSimulation,
        receptor_indices: list[int],
        max_num_steps: int,
        use_min_dist_rmsd_escape: bool = False,
        ) -> tuple[int, list[np.ndarray]]:
    simulation.counter = 0
    simulation.context.setTime(0.0)
    start_lig_prot_dist = simulation.RAMD_start()
    print(f"Starting ligand-protein distance: {np.linalg.norm(start_lig_prot_dist):.3} A")
    # Do the simulation steps and loop here. These are done every step
    successful_escape = False
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    ref_positions = state.getPositions(asNumpy=True)
    positions_list = [ref_positions]

    while simulation.counter < max_num_steps:
        lig_prot_distance = simulation.RAMD_step(simulation.ramd_steps)
        if lig_prot_distance * unit.angstroms > simulation.max_dist:
            simulation.max_distance_exceeded(simulation.counter)
            successful_escape = True
            break

    if not successful_escape:
        # Then we ran out of time
        simulation.max_time_exceeded(max_num_steps)
    else:
        # add the first and final frame
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        final_positions = state.getPositions(asNumpy=True)
        positions_list.append(final_positions)
        
    return simulation.counter, positions_list

def perform_ramd_simulation(
        simulation: ramd_simulation.RAMDSimulation,
        sim_parameters: dict,
        ramd_parameters: dict,
        ) -> list[np.ndarray]:
    time_step = sim_parameters["time_step"] * unit.picoseconds
    receptor_indices = ramd_parameters["receptor_indices"]
    max_num_steps = ramd_parameters["max_num_steps"]
    start_time = time.time()
    step_counter, positions = run_ramd_sim(
        simulation, receptor_indices, max_num_steps=max_num_steps, 
        use_min_dist_rmsd_escape=False)
    total_time = time.time() - start_time
    simulation_in_ns = step_counter * time_step.value_in_unit(unit.picoseconds) * 1e-3
    total_time_in_days = total_time / (86400.0)
    ns_per_day = simulation_in_ns / total_time_in_days
    print(f"simulation finished in {step_counter} steps")
    print("RAMD benchmark:", ns_per_day, "ns/day")
    return positions

def simulate_conventional(
        ff_parameters: dict,
        sim_parameters: dict,
        ramd_parameters: dict,
        num_structures_to_save: int,
        steps_between_structures: int,
        minimize: bool = True,
        cuda_device_index: str | None = "0",
        ):
    """
    Generate a number of structures from a conventional MD simulation, 
    which can be used as starting points for RAMD simulations. 
    """
    validate_sim_parameters(sim_parameters)
    validate_ramd_parameters(ramd_parameters)
    topology, system = choose_topology_system_from_params(ff_parameters, sim_parameters)
    integrator = choose_conventional_integrator(sim_parameters)
    platform, properties = choose_platform_and_properties(cuda_device_index)
    add_restraints_to_system(system, ff_parameters, sim_parameters)
    simulation = make_conventional_simulation(
        topology, system, integrator, platform, properties, ff_parameters,
        sim_parameters, minimize)
    structure_names = write_conformations_to_pdb(
        simulation, num_structures_to_save, steps_between_structures,
        structure_basename="starting_{}.pdb", 
        rec_indices=sim_parameters.get("restraint_indices", []),
    )
    return structure_names

def simulate_ramd(
        ff_parameters: dict,
        sim_parameters: dict,
        ramd_parameters: dict,
        starting_structure_name: str,
        minimize: bool,
        ramd_log_filename: str,
        cuda_device_index: str | None = "0",
        trajectory_filename: str | None = None
        ):
    """
    Run a RAMD simulation using the specified parameters.
    """
    validate_sim_parameters(sim_parameters)
    validate_ramd_parameters(ramd_parameters)
    topology, system = choose_topology_system_from_params(ff_parameters, sim_parameters)
    starting_structure = openmm_app.PDBFile(starting_structure_name)
    starting_positions = starting_structure.positions
    integrator = choose_ramd_integrator(
        sim_parameters, ramd_parameters, starting_positions)
    platform, properties = choose_platform_and_properties(cuda_device_index)
    add_restraints_to_system(system, ff_parameters, sim_parameters)
    simulation = make_ramd_simulation(
        topology, system, integrator, platform, properties, ff_parameters,
        sim_parameters, ramd_parameters, starting_structure_name, minimize,
        ramd_log_filename, trajectory_filename=trajectory_filename)
        
    positions = perform_ramd_simulation(
        simulation, sim_parameters, ramd_parameters)

    return positions
