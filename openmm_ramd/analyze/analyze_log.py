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
import PyGT

import openmm_ramd.analyze.parser as parser
import openmm_ramd.analyze.milestoning as milestoning
import openmm_ramd.analyze.remove_slant as remove_slant

kB_kcal_per_mol_per_kelvin = unit.MOLAR_GAS_CONSTANT_R.in_units_of(
    unit.kilocalories_per_mole / unit.kelvin)
A_per_nm = 10.0

MODE = 2
TINY_RATE = 1e-5

def make_xi_bins(num_xi_bins):
    xi_bins = []
    xi_values = []
    xi_bins_ranges = np.linspace(-1.0, 1.0, num_xi_bins+1)
    for j in range(num_xi_bins):
        xi_bins.append([])
        xi_value = 0.5 * (xi_bins_ranges[j] + xi_bins_ranges[j+1])
        xi_values.append(xi_value)
        
    return xi_bins, xi_values, xi_bins_ranges

#def compute_time(milestones, energy_profile_avg, diffusion_profile_avg, 
#                 beta, starting_index, bulk_milestone):
def compute_time(rate_matrix, starting_index, bulk_milestone):
    #rate_matrix = remove_slant.milestoning_statistics_from_profiles(
    #    milestones, energy_profile_avg, diffusion_profile_avg, beta)
    B, tau = PyGT.tools.load_CTMC(rate_matrix)
    mfpt_ij, mfpt_ji = PyGT.mfpt.compute_MFPT(
                    starting_index, bulk_milestone, B, tau)
    time_estimate = mfpt_ji
    return time_estimate

def correct_milestoning_model_for_ramd(rate_matrix_avg, transition_matrix_avg,
                                       avg_time_vector_avg):
    end_rate = rate_matrix_avg[-3, -2]
    rate_matrix_avg[-1, -2] = 0.5 * end_rate
    rate_matrix_avg[-3, -2] = 0.5 * end_rate
    rate_matrix_avg[-2, -2] = -end_rate
    rate_matrix_avg[-2, -1] = end_rate
    rate_matrix_avg[-1, -1] = -end_rate
    
    transition_matrix_avg[-1, -2] = 0.5
    transition_matrix_avg[-3, -2] = 0.5
    transition_matrix_avg[-2, -1] = 1.0
    
    avg_time_vector_avg[-1] = avg_time_vector_avg[-2]
    return

# TODO: remove all camel case
# TODO: break some of this code down into smaller functions
def analyze_log(ramd_log_file_list, num_milestones=10, num_xi_bins=5, 
                verbose=True, toy_mode=False):
    if verbose:
        print(f"Using {num_milestones} milestones.")
        print(f"Using {num_xi_bins} angle bins.")
    
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
    unitless_beta = beta.value_in_unit(unit.mole / unit.kilocalorie)
    
    if verbose:
        print("Number of trajectories extracted from log file(s):", 
              len(trajectory_list))
        print("forceOutFreq:", forceOutFreq)
        print("forceRAMD:", forceRAMD, "kcal/(mole * Angstrom)")
        print("timeStep:", timeStep, "ps")
        print("temperature:", temperature, "K")
        print("maxDist:", maxDist, "A")
        print("beta:", beta)
        #print("unitless beta:", unitless_beta)
    
    # Each trajectory represents a set of frames between force changes
    # Align the trajectories and convert to 1D, and xi values
    one_dim_trajs = parser.condense_trajectories(trajectory_list, onlyZ=toy_mode)
        
    #Construct a milestoning model based on trajectory fragments
    milestones, min_location, max_location, starting_cv_val, starting_index, h \
        = milestoning.uniform_milestone_locations(one_dim_trajs, num_milestones,
                                                  maxDist)
    bulk_milestone = num_milestones - 1
    # TODO: hacky
    print("milestones:", milestones, "h:", h)
    print("starting_index:", starting_index)
    xi_bins, xi_values, xi_bins_ranges = make_xi_bins(num_xi_bins)
        
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
    forceRAMD_in_nm_unitless = forceRAMD * A_per_nm
    forceRAMD_in_nm = forceRAMD_in_nm_unitless * unit.kilocalories_per_mole
    forceRAMD_in_A = forceRAMD * unit.kilocalories_per_mole
    
    # TODO: marked for removal
    #xi_escape_counts, xi_escape_blocks, xi_escape_block_rates \
    #    = parser.make_xi_escape_rates(xi_bins, maxDist)
    #xi_escape_times = xi_escape_blocks * frame_time
    #xi_escape_rates = xi_escape_block_rates * (1.0 / frame_time)
    
    unslanted_energy_profiles = np.zeros((num_xi_bins, num_milestones))
    diffusion_profiles = np.zeros((num_xi_bins, num_milestones))
    counts_profiles = np.zeros((num_xi_bins, num_milestones))
    
    rate_matrix_sum = np.zeros((num_milestones, num_milestones))
    transition_matrix_sum = np.zeros((num_milestones, num_milestones))
    avg_time_vector_sum = np.zeros((num_milestones, 1))
    counts_sum = np.zeros(num_milestones)
    for i in range(num_xi_bins):
        if verbose:
            print(f"num trajectories in bin {i}: {len(xi_bins[i])}")
        assert len(xi_bins[i]) > 0, "All angle bins need trajectories. "\
            "Either run more simulations or decrease the number of angle bins."
        xi = xi_values[i]
        xi_bin_trajs = xi_bins[i]
        count_matrix, time_vector, time_vector_up, time_vector_down, \
            transition_matrix, avg_time_vector, avg_time_vector_up, \
            avg_time_vector_down, rate_matrix \
            = milestoning.make_milestoning_model(
                xi_bin_trajs, milestones, frame_time, num_milestones,
                remove_unbalanced_frame_counts=False)
        slanting_force = xi * forceRAMD_in_nm_unitless
        
        if MODE == 1:
            energy_profile, diffusion_profile, counts \
                = remove_slant.recover_profiles(
                transition_matrix, time_vector, rate_matrix, count_matrix, 
                num_milestones, h, unitless_beta)
            unslanted_energy_profile = remove_slant.remove_slant(
                milestones, energy_profile, slanting_force)
            # milestoning statistics from profiles way
            unslanted_rate_matrix, unslanted_transition_matrix, \
                unslanted_time_vector \
                = remove_slant.milestoning_statistics_from_profiles(
                    milestones, unslanted_energy_profile, diffusion_profile, beta)
        elif MODE == 2:
            # Unslanting transition matrices way
            unslanted_rate_matrix, unslanted_transition_matrix, \
                unslanted_time_vector \
                    = remove_slant.unslant_transition_matrix_constant_approx(
                    milestones, transition_matrix, avg_time_vector, slanting_force,
                    beta)
            counts = milestoning.compute_counts(count_matrix, num_milestones)
            
        else:
            raise Exception(f"Mode not available: {MODE}")
        
        
        for j in range(num_milestones):
            # NOTE: is this really the best way? Should we average the
            #  probabilities and avg_times separately, then make a rate
            #  matrix? I don't think so.
            rate_matrix_sum[:, j] += counts[j] * unslanted_rate_matrix[:, j]
            transition_matrix_sum[:, j] += counts[j] * unslanted_transition_matrix[:, j]
            avg_time_vector_sum[j, 0] += counts[j] * unslanted_time_vector[j, 0]
            counts_sum[j] += counts[j]
        
        #unslanted_energy_profiles[i, :] = unslanted_energy_profile
        #diffusion_profiles[i, :] = diffusion_profile
        #counts_profiles[i, :] = counts
        #print("unslanted_transition_matrix")
        #print(unslanted_transition_matrix)
        #exit()
    
    rate_matrix_avg = np.zeros((num_milestones, num_milestones))
    transition_matrix_avg = np.zeros((num_milestones, num_milestones))
    avg_time_vector_avg = np.zeros((num_milestones, 1))
    for j in range(num_milestones):
        if counts_sum[j] == 0:
            rate_matrix_avg[j-1, j] = rate_matrix_avg[j, j-1]
            rate_matrix_avg[j, j] = -rate_matrix_avg[j, j-1]
        else:
            rate_matrix_avg[:, j] = rate_matrix_sum[:, j] / counts_sum[j]
            transition_matrix_avg[:, j] = transition_matrix_sum[:, j] / counts_sum[j]
        avg_time_vector_avg[j, 0] = avg_time_vector_sum[j, 0] / counts_sum[j]
    
    correct_milestoning_model_for_ramd(
        rate_matrix_avg, transition_matrix_avg, avg_time_vector_avg)
    
    print("transition_matrix_avg")
    print(transition_matrix_avg)
    
    averaged_stationary_probs = milestoning.compute_stationary_probabilities(
        transition_matrix_avg, avg_time_vector_avg, num_milestones, 
        starting_index, wrap_sink=False)
    
    averaged_energy_profile = remove_slant.make_energy_profile(
        averaged_stationary_probs, unitless_beta, starting_index)
    
    print("rate_matrix_avg")
    print(rate_matrix_avg)
    np.savetxt("/home/lvotapka/tmp/rate_matrix.txt", rate_matrix_avg)
    
    print("avg_time_vector_avg")
    print(avg_time_vector_avg)
    expected_time = h**2 / 2.0
    print("expected time per milestone:", h**2/2.0)
    
    #energy_profile_avg, diffusion_profile_avg \
    #    = remove_slant.average_unslanted_profiles(
    #    num_xi_bins, num_milestones, unitless_beta, unslanted_energy_profiles,
    #    diffusion_profiles, counts_profiles)
    #time_estimate = compute_time(
    #    milestones, energy_profile_avg, diffusion_profile_avg, beta, 
    #    starting_index, bulk_milestone)
    
    # Recompute main diagonal
    for j in range(num_milestones):
        rate_matrix_avg[j, j] = 0.0
        rate_matrix_avg[j, j] = -sum(rate_matrix_avg[:,j])
        
    time_estimate = compute_time(rate_matrix_avg, starting_index, bulk_milestone)
    plt.plot(milestones, averaged_energy_profile, "r", label="Recovered Potential energy $W(z) (kJ/mol)$")
    #plt.plot(milestones, avg_time_vector_avg, "b", label="Time vector")
    plt.ylim((-15.0, 10.0))
    plt.show()
    
    return time_estimate

# TODO: add Torque balancing in the RAMD.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "ramdLogFiles", metavar="RAMDLOGFILES", type=str, nargs="+",
        help="The RAMD log files to open and parse.")
    argparser.add_argument(
        "-n", "--numMilestones", dest="numMilestones", default=10,
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