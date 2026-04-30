"""
Run a RAMD simulation straight from an AMBER parm file and PDB file.
"""

import argparse
import time
from sys import stdout

import openmm.app as app
import openmm as mm
import openmm.unit as unit

import openmm_ramd.openmm_ramd as ramd
import openmm_ramd.integrator as ramd_integrator
import openmm_ramd.base as base

def run_amber_RAMD(pdb_filename, prmtop_filename, rec_indices, lig_indices, 
                   cuda_device_index=None, temperature=298.0, 
                   nonbonded_cutoff=1.0, target_pressure=None, time_step=0.002,
                   hydrogen_mass_amu=None,
                   ramd_force_magnitude=14.0, steps_per_RAMD_update=50,
                   RAMD_cutoff_distance=0.0025, RAMD_max_distance=1.5,
                   max_travel=9e9,
                   ramd_log_filename="ramd.log", minimize=False,
                   steps_per_energy_update=None, 
                   steps_per_trajectory_update=None, trajectory_filename=None,
                   max_num_steps=10000000, restraint_indices=None, 
                   restraint_force_constant=None, position_print_freq=None,
                   integrator_type="langevin", ramp_start_ns=0.0, ramp_force_increase_per_ns=0.0):
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    mypdb = app.PDBFile(pdb_filename)
    box_vectors = (mypdb.topology).getPeriodicBoxVectors()

    assert box_vectors is not None, "No box vectors "\
        "found in {}. ".format(pdb_filename) \
        + "Box vectors for an anchor must be defined with a CRYST "\
        "line within the PDB file."
    
    temperature = temperature * unit.kelvin
    nonbonded_cutoff = nonbonded_cutoff * unit.nanometers
    target_pressure = target_pressure * unit.bar
    time_step = time_step * unit.picoseconds
    ramd_force_magnitude = ramd_force_magnitude \
        * unit.kilocalories_per_mole / unit.angstrom
    RAMD_cutoff_distance = RAMD_cutoff_distance * unit.nanometers
    RAMD_max_distance = RAMD_max_distance * unit.nanometers
    max_travel = max_travel * unit.nanometers
    ramp_force_increase_per_ns = ramp_force_increase_per_ns * unit.kilocalories_per_mole / unit.angstrom

    if hydrogen_mass_amu:
        system = prmtop.createSystem(
            nonbondedMethod=app.PME, nonbondedCutoff=nonbonded_cutoff,
            constraints=app.HBonds, hydrogenMass=hydrogen_mass_amu*unit.amu)
    else:
        system = prmtop.createSystem(
            nonbondedMethod=app.PME, nonbondedCutoff=nonbonded_cutoff,
            constraints=app.HBonds)
        
    if target_pressure:
        barostat = mm.MonteCarloBarostat(target_pressure, temperature, 25)
        system.addForce(barostat)
    
    if integrator_type == "langevin":
        integrator = ramd_integrator.RamdLangevinIntegrator(
            temperature, 1/unit.picosecond, time_step, lig_indices, rec_indices, 
            mypdb.positions)
            
    elif integrator_type == "brownian":
        integrator = ramd_integrator.RamdBrownianIntegrator(
            temperature, 1/unit.picosecond, time_step, lig_indices, rec_indices, 
            mypdb.positions)
    else:
        raise Exception(f"Integrator type not supported: {integrator_type}")
    
    platform = mm.Platform.getPlatformByName('CUDA')
    if cuda_device_index is None:
        cuda_device_index = "0"
    properties = {"CudaDeviceIndex": cuda_device_index, "CudaPrecision": "mixed"}
    
    # Impose restraints
    print("restraining receptor atoms")
    if restraint_indices is None:
        restraint_indices = rec_indices
    
    if restraint_force_constant is None:
        restraint_force_constant = 1.0
    
    restraint_force_constant *= unit.kilocalories_per_mole / unit.angstrom**2
    
    print("restraint_indices:", restraint_indices)
    base.impose_receptor_restraints(
        system, mypdb.positions, restraint_force_constant, restraint_indices)
    
    if position_print_freq is None:
        position_print_freq = steps_per_RAMD_update

    ramd_steps = min(steps_per_RAMD_update, position_print_freq)
    
    simulation = ramd.RAMDSimulation(
        prmtop.topology, system, integrator, ramd_force_magnitude, lig_indices, 
        rec_indices, ramd_steps=ramd_steps, 
        r_min_ramd=RAMD_cutoff_distance, 
        force_out_freq=steps_per_RAMD_update, 
        max_dist=RAMD_max_distance, max_travel=max_travel,
        platform=platform, properties=properties, log_file_name=ramd_log_filename,
        position_print_freq=position_print_freq,
        ramp_start_ns=ramp_start_ns, ramp_force_increase_per_ns=ramp_force_increase_per_ns)
    
    simulation.context.setPositions(mypdb.positions)
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    if minimize:
        simulation.minimizeEnergy()
        
    simulation.context.setVelocitiesToTemperature(temperature)
    if steps_per_energy_update is not None:
        simulation.reporters.append(app.StateDataReporter(stdout, steps_per_energy_update, step=True,
                potentialEnergy=True, temperature=True, volume=True))
    if steps_per_trajectory_update is not None:
        assert trajectory_filename is not None, \
            "trajectory_filename must be assigned if "\
            "'steps_per_trajectory_update' is not None"
        dcd_reporter = app.DCDReporter(
            trajectory_filename, steps_per_trajectory_update)
        simulation.reporters.append(dcd_reporter)
    else:
        assert trajectory_filename is None, \
            "trajectory_filename cannot be assigned if "\
            "'steps_per_trajectory_update' is None"
        
    start_time = time.time()
    step_counter = simulation.run_RAMD_sim(max_num_steps=max_num_steps)
    
    total_time = time.time() - start_time
    simulation_in_ns = step_counter * time_step.value_in_unit(unit.picoseconds) * 1e-3
    total_time_in_days = total_time / (86400.0)
    ns_per_day = simulation_in_ns / total_time_in_days
    print("RAMD benchmark:", ns_per_day, "ns/day")
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "pdb_filename", metavar="PDB_FILENAME", type=str,
        help="The PDB file of the system to be simulated. The PDB file "\
        "defines the atomic positions of the system. It also must contain "\
        "box vector information in the form of a CRYST line.")
    argparser.add_argument(
        "prmtop_filename", metavar="PRMTOP_FILENAME", type=str, 
        help="The name of the forcefield file, this would be a .parm7 "\
        "or a .prmtop file.")
    
    argparser.add_argument(
        "receptor_indices", metavar="RECEPTOR_INDICES", type=str,
        help="A comma-separated list of integers defining site within the "\
        "pdb_filename structure. Ex: -r '1,2,0'.")
    argparser.add_argument(
        "-l", "--ligand_indices", dest="ligand_indices", 
        metavar="LIGAND_INDICES", type=str, default="",
        help="A comma-separated list of integers defining ligand within the "\
        "pdb_filename structure. Ex: -l '1,2,0'. Either the '-l' or '-L' "\
        "arguments must be included.")
    argparser.add_argument(
        "-L", "--ligand_resname", dest="ligand_resname", 
        metavar="LIGAND_RESNAME", type=str, default="",
        help="The residue name of the ligand molecule for automatic index "\
        "selection. Either the '-l' or '-L' arguments must be included.")
    
    argparser.add_argument(
        "-c", "--cuda_device_index", dest="cuda_device_index", default=None,
        help="modify which cuda_device_index to run the simulation on. For "\
        "example, the number 0 or 1 would suffice. To run on multiple GPU "\
        "indices, simply enter comma separated indices. Example: '0,1'. "\
        "If a value is not supplied, '0' will be used by default.", type=str)
    argparser.add_argument(
        "-T", "--temperature", dest="temperature", default=298.0,
        help="Temperature (in Kelvin). Default: 298.0.", type=float)
    argparser.add_argument(
        "-n", "--nonbonded_cutoff", dest="nonbonded_cutoff", default=1.0,
        help="Nonbonded cutoff (in nanometers). Default: 1.0.", type=float)
    argparser.add_argument(
        "-P", "--target_pressure", dest="target_pressure", default=None,
        help="Target pressure (in bar) for constant pressure simulations. "\
        "If set to None, simulation will not use constant pressure, but "\
        "rather constant volume. Default: None.", type=float)
    argparser.add_argument(
        "-t", "--time_step", dest="time_step", default=0.002,
        help="Time step to use for simulations (in picoseconds). "\
        "Default: 0.002.", type=float)
    argparser.add_argument(
        "-F", "--ramd_force_magnitude", dest="ramd_force_magnitude",
        default=14.0, help="The magnitude of the RAMD force used to push the "\
        "ligand (in kcal per mol per Angstrom). Default: 14.0.", type=float)
    argparser.add_argument(
        "-s", "--steps_per_ramd_update", dest="steps_per_ramd_update", 
        default=50, help="The number of steps to take between RAMD updates. "\
        "Default: 50.", type=int)
    argparser.add_argument(
        "-d", "--ramd_cutoff_distance", dest="ramd_cutoff_distance", 
        default=0.0025, help="The minimum distance that the ligand must "\
        "travel, relative to the receptor, otherwise the force is changed. "\
        "Default: 0.0025.", type=float)
    argparser.add_argument(
        "-D", "--ramd_max_distance", dest="ramd_max_distance", default=2.5,
        help="The distance that the ligand must move away from the receptor "\
        "in order for the simulation to end with an escape event. "\
        "Default: 2.5.", type=float)
    argparser.add_argument(
        "-R", "--ramd_log_filename", dest="ramd_log_filename", 
        default="ramd.log", help="The name of the log file to write RAMD "\
        "information for. This file is used to estimate the residence time "\
        "of the system. Default: ramd.log", type=str)
    argparser.add_argument(
        "-m", "--minimize", dest="minimize", default=False, 
        help="By default, minimizations of the system will not be performed. "\
        "Use this argument to activate minimization. Default: False.", 
        action="store_true")
    argparser.add_argument(
        "-e", "--steps_per_energy_update", dest="steps_per_energy_update", 
        default=None, help="The number of steps between reports to the system "\
        "energy, volume, and other information. If set to None, then this "\
        "information will not be printed. Default: None.", type=int)
    argparser.add_argument(
        "-u", "--steps_per_trajectory_update", 
        dest="steps_per_trajectory_update", default=None, 
        help="The number of steps between reports to the system "\
        "particle positions. If set to None, then this information will not "\
        "be reported. If not None, then the 'trajectory_filename' value "\
        "must be set. Default: None.", type=int)
    argparser.add_argument(
        "-U", "--trajectory_filename", dest="trajectory_filename", default=None, 
        help="The name of the trajectory file to be written if "\
        "'steps_per_trajectory_update' is not set to None. Default: None.", 
        type=int)
    argparser.add_argument(
        "-M", "--max_num_steps", dest="max_num_steps", default=10000000, 
        help="The number of steps, maximum, for the RAMD simulation. "\
        "Default: 10000000.", type=int)
    
    args = argparser.parse_args()
    args = vars(args)
    pdb_filename = args["pdb_filename"]
    prmtop_filename = args["prmtop_filename"]
    rec_indices = args["receptor_indices"]
    if rec_indices != "":
        rec_indices = base.initialize_indices(rec_indices)
    else:
        rec_indices = None
    lig_indices = args["ligand_indices"]
    if lig_indices != "":
        lig_indices = base.initialize_indices(lig_indices)
    else:
        lig_indices = None
    lig_resname = args["ligand_resname"]
    
    cuda_device_index = args["cuda_device_index"]
    temperature = args["temperature"]
    nonbonded_cutoff = args["nonbonded_cutoff"]
    target_pressure = args["target_pressure"]
    time_step = args["time_step"]
    ramd_force_magnitude = args["ramd_force_magnitude"]
    steps_per_ramd_update = args["steps_per_ramd_update"]
    ramd_cutoff_distance = args["ramd_cutoff_distance"]
    ramd_max_distance = args["ramd_max_distance"]
    ramd_log_filename = args["ramd_log_filename"]
    minimize = args["minimize"]
    steps_per_energy_update = args["steps_per_energy_update"]
    steps_per_trajectory_update = args["steps_per_trajectory_update"]
    trajectory_filename = args["trajectory_filename"]
    max_num_steps = args["max_num_steps"]
    
    # TODO: Handle ligand and receptor indices
    
    if lig_indices is None:
        assert lig_resname != "", \
            "If ligand_indices are not provided, ligand_resname must be."
        lig_indices = base.get_indices_ligand(
            pdb_filename=pdb_filename, lig_resname=lig_resname)
        
    else:
        assert lig_resname == "", \
            "If ligand_indices are provided, ligand_resname must not be."
    
    run_amber_RAMD(pdb_filename, prmtop_filename, rec_indices, lig_indices,
                   cuda_device_index, temperature=temperature, 
                   nonbonded_cutoff=nonbonded_cutoff, 
                   target_pressure=target_pressure, time_step=time_step, 
                   ramd_force_magnitude=ramd_force_magnitude, 
                   steps_per_RAMD_update=steps_per_ramd_update,
                   RAMD_cutoff_distance=ramd_cutoff_distance, 
                   RAMD_max_distance=ramd_max_distance,
                   ramd_log_filename=ramd_log_filename, minimize=minimize,
                   steps_per_energy_update=steps_per_energy_update, 
                   steps_per_trajectory_update=steps_per_trajectory_update, 
                   trajectory_filename=trajectory_filename,
                   max_num_steps=max_num_steps)
