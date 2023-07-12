"""
analyze_log.py

Read and parse a RAMD log file. Construct a series of milestoning models from 
the trajectory fragments for a set of bins of RAMD force vectors. The 
information from the milestoning models will be used in a sigma-RAMD model
that will allow one to expand the residence time in a Taylor series expansion
according to sigma-RAMD theory.
"""

import argparse

import numpy as np
from openmm import unit
import matplotlib.pyplot as plt

import openmm_ramd.analyze.parser as parser
import openmm_ramd.analyze.milestoning as milestoning
import openmm_ramd.analyze.sigma_ramd as sigma_ramd

kB_kcal_per_mol_per_kelvin = unit.MOLAR_GAS_CONSTANT_R.in_units_of(
    unit.kilocalories_per_mole / unit.kelvin)
A_per_nm = 10.0

# TODO: remove all camel case
# TODO: break some of this code down into smaller functions
def analyze_log(ramd_log_file_list, num_milestones=10, num_xi_bins=5, 
                verbose=True, toy_mode=False):
    if verbose:
        print(f"Using {num_milestones} milestones.")
        print(f"Using {num_xi_bins} angle bins.")
    xi_bins_ranges = np.linspace(-1.0, 1.0, num_xi_bins+1)
    trajectory_list = []
    forceOutFreq = None
    forceRAMD = None
    timeStep = None
    temperature = None
    maxDist = None
    for ramd_log_filename in ramd_log_file_list:
        trajectories, forceOutFreq_, forceRAMD_, timeStep_, num_trajectories_, \
            temperature_, maxDist_ = parser.parse_ramd_log_file(
                ramd_log_filename)
        if forceOutFreq is None:
            forceOutFreq = forceOutFreq_
            forceRAMD = forceRAMD_
            timeStep = timeStep_
            temperature = temperature_
            maxDist = maxDist_
        else:
            assert forceOutFreq == forceOutFreq_, "RAMD logs contain differing forceOutFreq values."
            assert forceRAMD == forceRAMD_, "RAMD logs contain differing forceRAMD values."
            assert timeStep == timeStep_, "RAMD logs contain differing timeStep values."
            assert temperature == temperature_, "RAMD logs contain differing temperature values."
            assert maxDist == maxDist_, "RAMD logs contain differing maxDist values."
            
        trajectory_list += trajectories
    
    beta = 1.0 / (temperature * unit.kelvin * kB_kcal_per_mol_per_kelvin)
    
    if verbose:
        print("Number of trajectories extracted from log file(s):", 
              len(trajectory_list))
        print("forceOutFreq:", forceOutFreq)
        print("forceRAMD:", forceRAMD, "kcal/(mole * Angstrom)")
        print("timeStep:", timeStep, "ps")
        print("temperature:", temperature, "K")
        print("maxDist:", maxDist, "A")
        print("beta:", beta)
    
    # Each trajectory represents a set of frames between force changes
    # Align the trajectories and convert to 1D, and xi values
    one_dim_trajs = parser.condense_trajectories(trajectories, onlyZ=toy_mode)
    
    #Construct a milestoning model based on trajectory fragments
    milestones, min_location, max_location, starting_cv_val, starting_index \
        = milestoning.uniform_milestone_locations(one_dim_trajs, num_milestones,
                                                  maxDist)
    print("milestones:", milestones)
    xi_bins = []
    for j in range(num_xi_bins):
        xi_bins.append([])
        
    for i, trajectory in enumerate(one_dim_trajs):
        traj_xi = trajectory[0][1]
        found_bin = False
        for j in range(num_xi_bins):
            if (traj_xi >= xi_bins_ranges[j]) \
                    and (traj_xi <= xi_bins_ranges[j+1]):
                xi_bins[j].append(trajectory)
                found_bin = True
                break
        
        if not found_bin:
            raise Exception(f"xi value {traj_xi} not placed in bin.")
    
    frame_time = timeStep * forceOutFreq
    xi_bin_time_profiles = np.zeros((num_xi_bins, num_milestones, num_milestones))
    xi_bin_times = np.zeros(num_xi_bins)
    for i in range(num_xi_bins):
        if verbose:
            print(f"num trajectories in bin {i}: {len(xi_bins[i])}")
        xi_bin_trajs = xi_bins[i]
        count_matrix, time_vector, time_vector_up, time_vector_down, \
            transition_matrix, avg_time_vector, avg_time_vector_up, \
            avg_time_vector_down, rate_matrix \
            = milestoning.make_milestoning_model(
                xi_bin_trajs, milestones, frame_time, num_milestones)
        
        xi_bin_times[i] = milestoning.compute_residence_time(
            transition_matrix, avg_time_vector, num_milestones, starting_index)
        time_profile_matrix = milestoning.make_time_profiles(
            transition_matrix, avg_time_vector, avg_time_vector_up, 
            avg_time_vector_down, num_milestones)
        xi_bin_time_profiles[i,:,:] = time_profile_matrix[:,:]
        
    # Use sigma-RAMD to analyze
    
    forceRAMD_in_nm = forceRAMD * A_per_nm
    forceRAMD_in_nm = forceRAMD_in_nm * unit.kilocalories_per_mole
    forceRAMD = forceRAMD * unit.kilocalories_per_mole
    calc = sigma_ramd.functional_expansion_1d_ramd(
        xi_bin_time_profiles, xi_bin_times, force_constant=forceRAMD_in_nm, beta=beta,
        min_cv_val=min_location, max_cv_val=max_location, 
        starting_cv_val=starting_cv_val)
    time_estimate = calc.time_estimate_taylor_series_second_order()
    return time_estimate

# TODO: add Torque balancing in the RAMD.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "ramdLogFiles", metavar="RAMDLOGFILES", type=str, nargs="+",
        help="The RAMD log files to open and parse.")
    argparser.add_argument(
        "-n", "--numMilestones", dest="numMilestones", default=7,
        help="The number of milestones to use in the milestoning "\
        "calculation.", type=int)
    argparser.add_argument(
        "-a", "--numAngleBins", dest="numAngleBins", default=10,
        help="The number of bins to divide the force vectors into.", type=int)
    argparser.add_argument(
        "-t", "--toyMode", dest="toyMode", default=False,
        help="In toy mode, we assume that the main CV of the simulation is "\
        "the Z-axis of the zeroth particle. In contrast, the CV of a non-"
        "toy system is found using a more complex procedure.", 
        action="store_true")
    
    args = argparser.parse_args()
    args = vars(args)
    ramdLogFiles = args["ramdLogFiles"]
    numMilestones = args["numMilestones"]
    numAngleBins = args["numAngleBins"]
    toyMode = args["toyMode"]
    
    time_estimate = analyze_log(ramdLogFiles, numMilestones, numAngleBins,
                                toy_mode=toyMode)
    print("Estimated unbiased time:", time_estimate)