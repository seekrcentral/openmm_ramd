"""
Run a RAMD simulation straight from an AMBER parm file and PDB file.
"""

import argparse
import openmm.unit as unit

import openmm_ramd.helper as helper
import openmm_ramd.base as base

def run_amber_RAMD(pdb_filename, prmtop_filename, rec_indices, lig_indices, 
                   cuda_device_index=None, temperature=300.0,
                   nonbonded_cutoff=1.0, target_pressure=None, time_step=0.002,
                   hydrogen_mass_amu=None,
                   ramd_force_magnitude=14.0, steps_per_RAMD_update=50,
                   RAMD_cutoff_distance=0.0025, RAMD_max_distance=3.0,
                   max_travel=9e9,
                   ramd_log_filename="ramd.log", minimize=False,
                   steps_per_energy_update=25000,
                   steps_per_trajectory_update=None, trajectory_filename=None,
                   max_num_steps=20000000, restraint_indices=None,
                   restraint_force_constant=None, position_print_freq=50,
                   integrator_type="langevin", ramp_start_ns=0.0, ramp_force_increase_per_ns=0.0,
                   *, friction_coeff=1.0, random_seed=0, collision_frequency=1.0,
                   chain_length=3, num_mts=3, num_yoshida_suzuki=7,
                   starting_state_name=None):
    if cuda_device_index is None:
        cuda_device_index = "0"
    restraint_strength = None
    if restraint_force_constant is not None:
        restraint_strength = (
            restraint_force_constant * unit.kilocalories_per_mole / unit.angstrom**2
        ).value_in_unit(unit.kilojoules_per_mole / unit.nanometer**2)
    ff_parameters = {
        "type": "amber",
        "prmtop_filename": prmtop_filename,
        "pdb_filename": pdb_filename,
    }
    sim_parameters = {
        "temperature": temperature,
        "nonbonded_cutoff": nonbonded_cutoff,
        "pressure": target_pressure,
        "time_step": time_step,
        "hydrogen_mass": hydrogen_mass_amu or None,
        "integrator_type": integrator_type,
        "friction_coeff": friction_coeff,
        "random_seed": random_seed,
        "collision_frequency": collision_frequency,
        "chain_length": chain_length,
        "num_mts": num_mts,
        "num_yoshida_suzuki": num_yoshida_suzuki,
        "restraint_indices": restraint_indices,
        "restraint_force_constant": restraint_strength,
        "steps_per_energy_update": steps_per_energy_update,
        "steps_per_trajectory_update": steps_per_trajectory_update,
    }
    ramd_parameters = {
        "receptor_indices": rec_indices,
        "ligand_indices": lig_indices,
        "ramd_force_magnitude": ramd_force_magnitude,
        "steps_per_RAMD_update": steps_per_RAMD_update,
        "RAMD_cutoff_distance": RAMD_cutoff_distance,
        "RAMD_max_distance": RAMD_max_distance,
        "max_travel": max_travel,
        "max_num_steps": max_num_steps,
        "position_print_freq": position_print_freq,
        "ramp_start_ns": ramp_start_ns,
        "ramp_force_increase_per_ns": ramp_force_increase_per_ns,
    }
    helper.simulate_ramd(
        ff_parameters=ff_parameters,
        sim_parameters=sim_parameters,
        ramd_parameters=ramd_parameters,
        starting_structure_name=pdb_filename,
        minimize=minimize,
        ramd_log_filename=ramd_log_filename,
        cuda_device_index=cuda_device_index,
        trajectory_filename=trajectory_filename,
        starting_state_name=starting_state_name,
    )

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
        "receptor_indices", metavar="RECEPTOR_INDICES", type=str, nargs="?",
        help="Comma-separated receptor atom indices; alternatively use -r/--receptor_indices.")
    argparser.add_argument(
        "-r", "--receptor_indices", dest="receptor_indices_option",
        metavar="RECEPTOR_INDICES", type=str,
        help="A comma-separated list of receptor atom indices. Ex: -r '1,2,0'.")
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
        "-T", "--temperature", dest="temperature", default=300.0,
        help="Temperature (in Kelvin). Default: 300.0.", type=float)
    argparser.add_argument(
        "-n", "--nonbonded_cutoff", dest="nonbonded_cutoff", default=1.0,
        help="Nonbonded cutoff (in nanometers). Default: 1.0.", type=float)
    argparser.add_argument(
        "-P", "--target_pressure", dest="target_pressure", default=None,
        help="Target pressure (in bar). Use 'none' for constant volume. Default: None.",
        type=lambda value: None if value.lower() == "none" else float(value))
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
        "-D", "--ramd_max_distance", dest="ramd_max_distance", default=3.0,
        help="The distance that the ligand must move away from the receptor "\
        "in order for the simulation to end with an escape event. "\
        "Default: 3.0.", type=float)
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
        default=25000, help="Steps between energy/volume reports. "
        "Use 'none' to disable. Default: 25000.",
        type=lambda value: None if value.lower() == "none" else int(value))
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
        type=str)
    argparser.add_argument(
        "-M", "--max_num_steps", dest="max_num_steps", default=20000000,
        help="The number of steps, maximum, for the RAMD simulation. "\
        "Default: 20000000.", type=int)
    
    argparser.add_argument(
        "--integrator_type", default="langevin",
        choices=["langevin", "brownian", "verlet", "nose_hoover"],
        help="RAMD integrator. Default: langevin.")
    argparser.add_argument(
        "--friction_coeff", type=float, default=1.0,
        help="Langevin/Brownian friction (1/ps). Default: 1.0.")
    argparser.add_argument(
        "--random_seed", type=int, default=0,
        help="Seed for velocity initialization and stochastic integration; 0 lets OpenMM choose.")
    argparser.add_argument(
        "--collision_frequency", type=float, default=1.0,
        help="Nose-Hoover collision frequency (1/ps). Default: 1.0.")
    argparser.add_argument(
        "--chain_length", type=int, default=3,
        help="Nose-Hoover chain length. Default: 3.")
    argparser.add_argument(
        "--num_mts", type=int, default=3,
        help="Nose-Hoover thermostat substeps. Default: 3.")
    argparser.add_argument(
        "--num_yoshida_suzuki", type=int, default=7,
        help="Nose-Hoover Yoshida-Suzuki terms. Default: 7.")
    argparser.add_argument(
        "--hydrogen_mass_amu", type=float, default=None,
        help="Hydrogen mass for mass repartitioning (daltons). Default: unchanged.")
    argparser.add_argument(
        "--starting_state_name", default=None,
        help="OpenMM State XML with positions, velocities, and box vectors; use a matching PDB.")
    argparser.add_argument(
        "--restraint_indices", default=None,
        help="Comma-separated restrained atom indices. Default: no restraints.")
    argparser.add_argument(
        "--restraint_force_constant", type=float, default=None,
        help="Restraint strength (kcal/mol/angstrom**2); required when restraint indices are supplied.")
    argparser.add_argument(
        "--position_print_freq", type=int, default=50,
        help="Steps between COM position log entries. Default: 50.")
    argparser.add_argument(
        "--max_travel", type=float, default=9e9,
        help="Maximum travel before changing force direction (nm). Default: 9e9.")
    argparser.add_argument(
        "--ramp_start_ns", type=float, default=0.0,
        help="Start time for force ramping (ns). Default: 0.0.")
    argparser.add_argument(
        "--ramp_force_increase_per_ns", type=float, default=0.0,
        help="Force increase in (kcal/mol/angstrom)/ns. Default: 0.0 (constant force).")

    args = argparser.parse_args()
    args = vars(args)
    pdb_filename = args["pdb_filename"]
    prmtop_filename = args["prmtop_filename"]
    rec_indices = args["receptor_indices"]
    receptor_indices_option = args["receptor_indices_option"]
    if rec_indices is not None and receptor_indices_option is not None:
        argparser.error("Supply receptor indices either positionally or with -r/--receptor_indices, not both.")
    if receptor_indices_option is not None:
        rec_indices = receptor_indices_option
    if rec_indices is None:
        argparser.error("Receptor indices are required, positionally or with -r/--receptor_indices.")
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
    
    restraint_indices = args["restraint_indices"]
    if restraint_indices is not None:
        restraint_indices = (base.initialize_indices(restraint_indices)
                             if restraint_indices else [])
    
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
                   max_num_steps=max_num_steps,
                   integrator_type=args["integrator_type"],
                   friction_coeff=args["friction_coeff"],
                   random_seed=args["random_seed"],
                   collision_frequency=args["collision_frequency"],
                   chain_length=args["chain_length"],
                   num_mts=args["num_mts"],
                   num_yoshida_suzuki=args["num_yoshida_suzuki"],
                   hydrogen_mass_amu=args["hydrogen_mass_amu"],
                   starting_state_name=args["starting_state_name"],
                   restraint_indices=restraint_indices,
                   restraint_force_constant=args["restraint_force_constant"],
                   position_print_freq=args["position_print_freq"],
                   max_travel=args["max_travel"],
                   ramp_start_ns=args["ramp_start_ns"],
                   ramp_force_increase_per_ns=args["ramp_force_increase_per_ns"])
