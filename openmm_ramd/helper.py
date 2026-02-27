"""
A series of helper function for preparing and running RAMD simulations.

Includes:
 - Functions for creating systems and simulation objects from different 
   input file types.
"""

import sys
import os

import openmm
import openmm.app as openmm_app
import openmm.unit as unit

import openmm_ramd.base as openmm_ramd_base

ALLOWED_SIM_PARAMETER_KEYS = {
    "temperature",
    "time_step",
    "nonbonded_cutoff",
    "pressure",
    "friction_coeff",
    "random_seed",
    "integrator_type",
    "restraint_force_constant",
    "restraint_indices",
}

ALLOWED_RAMD_PARAMETER_KEYS = {
    "...",
}

def validate_sim_parameters(sim_parameters: dict):
    for sim_param_key in sim_parameters.keys():
        assert sim_param_key in ALLOWED_SIM_PARAMETER_KEYS, \
            f"sim_parameters should only have keys {ALLOWED_SIM_PARAMETER_KEYS}"

def validate_ramd_parameters(ramd_parameters: dict):
    for ramd_param_key in ramd_parameters.keys():
        assert ramd_param_key in ALLOWED_RAMD_PARAMETER_KEYS, \
            f"ramd_parameters should only have keys {ALLOWED_RAMD_PARAMETER_KEYS}"

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
    system = prmtop.createSystem(
        nonbondedMethod=openmm_app.PME, nonbondedCutoff=nonbonded_cutoff,
        constraints=openmm_app.HBonds, hydrogenMass=3.0*unit.daltons,)
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
        steps_per_energy_update: int | None = None,
        ) -> openmm.app.Simulation:
    simulation = openmm_app.Simulation(
        topology, system, integrator, platform, properties)
    pdb = openmm_app.PDBFile(ff_parameters["pdb_filename"])
    simulation.context.setPositions(pdb.positions)
    box_vectors = topology.getPeriodicBoxVectors()
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    if minimize:
        simulation.minimizeEnergy()
    temperature = sim_parameters["temperature"] * unit.kelvin
    random_seed = sim_parameters.get("random_seed", 0)
    simulation.context.setVelocitiesToTemperature(temperature, random_seed)
    if steps_per_energy_update is not None:
        simulation.reporters.append(openmm_app.StateDataReporter(
            sys.stdout, steps_per_energy_update, step=True,
                potentialEnergy=True, temperature=True, volume=True))
    return simulation

# STOPPED HERE
def write_conformations_to_pdb(
        simulation: openmm.app.Simulation,
        num_structures_to_save: int,
        steps_between_structures: int,
        structure_basename: str,
        rec_indices: list[int],
        ) -> list[str]:
    structure_names = []
    traj = mdtraj.load()
    for i in range(num_structures_to_save):
        simulation.step(steps_between_structures)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        ref_positions = state.getPositions(asNumpy=True, )
        positions_list = [ref_positions]
        traj.xyz = np.array(positions_list)
        atom_set_list = []
        for index in rec_indices:
            atom_set_list.append(set([traj.topology.atom(index)]))
        assert len(atom_set_list) > 0
        traj.image_molecules(inplace=True, anchor_molecules=atom_set_list)
        structure_fullname = structure_basename+f"{i}.pdb"
        print("saving structure to", structure_fullname)
        traj.save(os.path.join(structure_fullname))
        structure_names.append(structure_fullname)

    return structure_names

def simulate_conventional(
        ff_parameters: dict,
        sim_parameters: dict,
        ramd_parameters: dict,
        num_structures_to_save: int,
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