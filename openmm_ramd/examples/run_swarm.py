"""
Run a swarm of RAMD simulations starting from a set of conformations
generated from a conventional MD simulation.
"""

import os
import glob
import json
import argparse

import numpy as np
import mdtraj

import openmm_ramd.helper as openmm_ramd_helper

DEFAULT_STEPS_BETWEEN_STRUCTURES = 2500000

def run_swarm(
        parameter_file: str,
        cuda_device_index: str | None = "0",
        use_xml_restarts: bool = False,
        ):
    with open(parameter_file) as f:
        parameters = json.load(f)

    ff_parameters = parameters.get("ff_parameters", {})
    sim_parameters = parameters.get("sim_parameters", {})
    ramd_parameters = parameters.get("ramd_parameters", {})

    num_starting_structures = ramd_parameters.get("num_starting_structures", 5)
    steps_between_structures = ramd_parameters.get("steps_between_structures", 
                                                   DEFAULT_STEPS_BETWEEN_STRUCTURES)
    num_velocity_restarts = ramd_parameters.get("num_velocity_restarts", 15)

    if use_xml_restarts:
        from openmm_ramd.examples.run_swarm_fixed import ensure_starting_structures

        ensure_starting_structures(
            ff_parameters, sim_parameters, ramd_parameters,
            num_starting_structures, steps_between_structures,
            cuda_device_index=cuda_device_index)
        structure_names = [f"starting_{i}.pdb" for i in range(num_starting_structures)]
    elif num_starting_structures > 0:
        starting_names_glob = os.path.join("starting_*.pdb")
        if len(glob.glob(starting_names_glob)) >= num_starting_structures:
            print("Starting structures already exist - not regenerating.")
            structure_names = sorted(glob.glob(starting_names_glob))
        else:
            structure_names = openmm_ramd_helper.simulate_conventional(
                ff_parameters=ff_parameters,
                sim_parameters=sim_parameters,
                ramd_parameters=ramd_parameters,
                num_structures_to_save=num_starting_structures,
                steps_between_structures=steps_between_structures,
                minimize=True,
                cuda_device_index=cuda_device_index,
                )
        print("Finished conventional sims")
    else:
        structure_names = [ff_parameters["pdb_filename"]]
    
    print(f"Starting at RAMD simulation 0.")
    for i, structure_name in enumerate(structure_names):
        for j in range(num_velocity_restarts):
            # TODO: add i and j to random seed calc.
            ramd_log_filename = f"ramd_{i}_{j}.log"
            trajectory_filename = None
            if sim_parameters.get("steps_per_trajectory_update") is not None:
                trajectory_filename = f"ramd_trajectory_{i}_{j}.dcd"
            if openmm_ramd_helper.should_run(ramd_log_filename):
                positions = openmm_ramd_helper.simulate_ramd(
                    ff_parameters=ff_parameters,
                    sim_parameters=sim_parameters,
                    ramd_parameters=ramd_parameters,
                    starting_structure_name=structure_name,
                    minimize=False,
                    ramd_log_filename=ramd_log_filename,
                    cuda_device_index=cuda_device_index,
                    trajectory_filename=trajectory_filename,
                    starting_state_name=f"starting_{i}.xml" if use_xml_restarts else None,
                    )
                # Save the frames
                pdb_filename = ff_parameters["pdb_filename"]
                traj = mdtraj.load(pdb_filename)
                traj.xyz = np.array([positions[0]])
                frame_traj = mdtraj.load(pdb_filename)
                for k in range(1, len(positions)):
                    frame_traj.xyz = np.array([positions[k]])
                    traj = traj.join(frame_traj)
                receptor_indices = ramd_parameters["receptor_indices"]
                atom_set_list = [set(traj.topology.atom(i) for i in receptor_indices)]
                # Convert to float32 to avoid dtype mismatch with mdtraj's C extension
                traj.xyz = traj.xyz.astype(np.float32)
                traj.image_molecules(inplace=True, anchor_molecules=atom_set_list)
                traj.save(f"ramd_frames_{i}_{j}.dcd")
                if i < num_starting_structures - 1 or j < num_velocity_restarts -1:
                    print("Completed simulation, now on to simulation", i+1, j+1)
            else:
                print(f"Skipping simulation {i} {j} with output file "
                    f"{ramd_log_filename}. Already completed.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "parameter_file", metavar="PARAMETER_FILE", type=str,
        help="The parameter file containing all the input files for a system.")

    argparser.add_argument(
        "-c", "--cuda_device_index", dest="cuda_device_index", default="0",
        help="modify which cuda_device_index to run the simulation on. For "\
        "example, the number 0 or 1 would suffice. To run on multiple GPU "\
        "indices, simply enter comma separated indices. Example: '0,1'. "\
        "If a value is not supplied, '0' will be used by default.", type=str)
    
    argparser.add_argument(
        "-u", "--use-xml-restarts", action="store_true",
        help="Use saved PDB/XML replica states, generating missing pairs with "
        "separate conventional MD runs.")

    args = argparser.parse_args()
    args = vars(args)
    parameter_file = args["parameter_file"]
    cuda_device_index = args["cuda_device_index"]
    run_swarm(parameter_file, cuda_device_index, args["use_xml_restarts"])
