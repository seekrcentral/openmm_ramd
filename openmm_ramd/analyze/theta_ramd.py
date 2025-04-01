"""
analyze_log.py

Read and parse a RAMD log file. Construct a series of milestoning models from 
the trajectory fragments for a set of bins of RAMD force vectors. The 
information from the milestoning models will be used in a theta-RAMD model
that will allow one to obtain an estimate of the absolute residence time 
according to theta-RAMD theory.
"""

import os
import glob
import shutil
import argparse

import numpy as np
from openmm import unit
import matplotlib.pyplot as plt

import openmm_ramd.analyze.modules.parser as parser
import openmm_ramd.analyze.modules.samples as samples
import openmm_ramd.analyze.modules.milestoning as milestoning

A_per_nm = 10.0
CONV_PRINT_UPDATE_INTERVAL = 5
SEEK_FROM_END_SIZE = 1000000
STRIDE_TO_FIND_CV = 10
CONVERGE_NUM_ERROR_SAMPLES = 0 # 10?

class InsufficientTrajectoriesError(Exception):
    pass

def make_xi_bin_plots(milestones, xi_bins, image_directory=None):
    YLIM_MIN_BOUND = -20.0
    YLIM_MAX_BOUND = 20.0
    
    # First plot the slanted profiles
    pi_fig1, ax = plt.subplots()
    global_ylim_min = YLIM_MAX_BOUND
    global_ylim_max = YLIM_MIN_BOUND
    for i, xi_bin in enumerate(xi_bins):
        xi = xi_bin.midpoint
        ylim_min = np.max((np.min(xi_bin.main_data_sample.free_energy_profile), YLIM_MIN_BOUND))
        if ylim_min < global_ylim_min:
            global_ylim_min = ylim_min
            
        ylim_max = np.min((np.max(xi_bin.main_data_sample.free_energy_profile), YLIM_MAX_BOUND))
        if ylim_max > global_ylim_max:
            global_ylim_max = ylim_max
        plt.plot(milestones, xi_bin.main_data_sample.free_energy_profile, label=f"$\\xi$={xi:.2f}")
        
    plt.ylim((global_ylim_min, global_ylim_max))
    #plt.legend()
    plt.ylabel("$W(z)$ (kJ/mol)")
    plt.xlabel("$z (\\AA)$")
    plt.title("Slanted Free Energy Profiles")
    
    if image_directory is None:
        plt.show()
    else:
        plt.savefig(os.path.join(image_directory, "slanted_profile.png"))
    
    plt.close()
    
    # Next, plot the unslanted profiles
    pi_fig2, ax = plt.subplots()
    global_ylim_min = YLIM_MAX_BOUND
    global_ylim_max = YLIM_MIN_BOUND
    for i, xi_bin in enumerate(xi_bins):
        xi = xi_bin.midpoint
        ylim_min = np.max((np.min(xi_bin.main_data_sample.free_energy_profile), YLIM_MIN_BOUND))
        if ylim_min < global_ylim_min:
            global_ylim_min = ylim_min
        
        ylim_max = np.min((np.max(xi_bin.main_data_sample.free_energy_profile), YLIM_MAX_BOUND))
        if ylim_max > global_ylim_max:
            global_ylim_max = ylim_max
        plt.plot(milestones, xi_bin.main_data_sample_unslanted.free_energy_profile, label=f"$\\xi$={xi:.2f}")
        
    plt.ylim((global_ylim_min, global_ylim_max))
    #plt.legend()
    plt.ylabel("$W(z)$ (kJ/mol)")
    plt.xlabel("$z (\\AA)$")
    plt.title("Unslanted Free Energy Profiles")
    if image_directory is None:
        plt.show()
    else:
        plt.savefig(os.path.join(image_directory, "unslanted_profile.png"))
    
    plt.close()
    return

def analyze_traj_set(
        xi_bins, starting_index, num_milestones, milestones, h, situation, 
        bulk_milestones, beta, unitless_beta, forceRAMD, num_error_samples, 
        flatten_beyond_zero_count, stride_error_samples, skip_error_samples, 
        verbose=False, debug_directory=None):
    # Combine stats into Xi_bins
    for i, xi_bin in enumerate(xi_bins):
        count_matrix = xi_bin.count_matrix
        time_vector = xi_bin.time_vector
        time_matrix = xi_bin.time_matrix
        avg_time_vector, avg_time_matrix\
            = milestoning.make_avg_milestone_statistics(
                count_matrix, time_vector, time_matrix)
        if debug_directory is not None:
            count_matrix_filename = os.path.join(
                debug_directory, f"count_matrix_{i}.txt")
            np.savetxt(count_matrix_filename, count_matrix, fmt="%4d")
            time_vector_filename = os.path.join(
                debug_directory, f"time_vector_{i}.txt")
            np.savetxt(time_vector_filename, time_vector, fmt="%.3e")
            avg_time_vector_filename = os.path.join(
                debug_directory, f"avg_time_vector_{i}.txt")
            np.savetxt(avg_time_vector_filename, avg_time_vector, fmt="%.3e")
            time_matrix_filename = os.path.join(
                debug_directory, f"time_matrix_{i}.txt")
            np.savetxt(time_matrix_filename, time_matrix, fmt="%.3e")
        
        if verbose:
            print(f"count_matrix {i}:")
            print(count_matrix)
            print(f"time_vector {i}:")
            print(time_vector)
        
        xi_bin.assign_main_data_sample(
            count_matrix, time_vector, avg_time_vector, avg_time_matrix)
        xi_bin.make_profiles(starting_index, wrap_sink=False, 
                                     unitless_beta=unitless_beta)
        xi_bin.unslant_data_samples(forceRAMD, beta)
        xi_bin.counts = milestoning.compute_counts(count_matrix, num_milestones)
        xi_bin.make_profiles(starting_index, wrap_sink=False, 
                             unitless_beta=unitless_beta, 
                             data_sample=xi_bin.main_data_sample_unslanted)
        if debug_directory is not None:
            counts_vector_filename = os.path.join(
                debug_directory, f"counts_{i}.txt")
            np.savetxt(counts_vector_filename, xi_bin.counts, fmt="%d")
            unslanted_energy_profile_filename = os.path.join(
                debug_directory, f"unslanted_free_energy_{i}.txt")
            np.savetxt(unslanted_energy_profile_filename, 
                       xi_bin.main_data_sample_unslanted.free_energy_profile,
                       fmt="%.3e")
        
    analysis = samples.Analysis(num_milestones, milestones, h, situation, 
                                bulk_milestones, verbose, debug_directory)
    analysis.avg_unslanted_xi_bins(xi_bins)
    if flatten_beyond_zero_count:
        analysis.estimate_matrix_beyond_zero_count(starting_index)
    
    if num_error_samples > 0:
        analysis.sample_error_matrices(
            num_error_samples, stride_error_samples, skip_error_samples)
    analysis.make_profiles(starting_index, wrap_sink=False, 
                           unitless_beta=unitless_beta)
    analysis.make_times(starting_index)
    return analysis

def theta_ramd(ramd_log_file_list, num_milestones=20, num_angle_bins=10, 
                toy_mode=False, num_error_samples=100, 
                stride_error_samples=None, skip_error_samples=None, 
                cut_out_drift=False, image_directory=None, verbose=False,
                display_xi_bin_plots=False, flatten_beyond_zero_count=True,
                debug_directory=None, trajectory_stride=1, convergence=False):
    if debug_directory is not None:
        if not os.path.exists(debug_directory):
            os.mkdir(debug_directory)
        else:
            for filename in os.listdir(debug_directory):
                full_path = os.path.join(debug_directory, filename)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)
        
    if verbose:
        print(f"Using {num_milestones} milestones.")
        print(f"Using {num_angle_bins} angle bins.")
    
    
    force_out_freq = None
    position_print_freq = None
    forceRAMD = None
    time_step = None
    temperature = None
    max_dist = None
    num_total_exits = 0
    num_total_trajs = 0
    num_total_log_files = len(ramd_log_file_list)
    milestones = None
    count_matrix = None
    time_vector = None
    time_matrix = None
    xi_bins = None
    converge_list = []
    if verbose:
        print("Now loading log files...")
    
    print("about to read settings")
    # Read settings here first
    _, force_out_freq, position_print_freq, forceRAMD, time_step, _, \
        temperature, max_dist, _, _, _ \
                = parser.parse_ramd_log_file(ramd_log_file_list[0], 
                                             read_only_settings=True)
    print("settings read")
    # Read enough of the last frames just to get the CV
    if toy_mode:
        norm_avgx = 0.0
        norm_avgy = 0.0
        norm_avgz = 1.0
        norm_avg_vec = np.array([0.0, 0.0, 1.0])
        
    else:
        # First, parse the last few frames of each log file to determine the CV
        end_traj_list = []
        for ramd_log_filename in ramd_log_file_list:
            end_trajs, force_out_freq_, position_print_freq_, forceRAMD_, \
                time_step_, num_trajectories_, temperature_, max_dist_, \
                num_exits, trajectories_file_indices, trajectories_starting_step\
                    = parser.parse_ramd_log_file(
                        ramd_log_filename, stride=STRIDE_TO_FIND_CV,
                        skip_checks=True)
            end_traj_list += end_trajs
        
        #norm_avg_vec, norm_avgx, norm_avgy, norm_avgz \
        #    = parser.find_prin_comp_cv(
        #        end_traj_list, max_dist, debug_directory=None)
        norm_avg_vec, norm_avgx, norm_avgy, norm_avgz \
            = parser.get_cv_vec_traj_shells(end_traj_list)
    
    print("CV chosen: ", norm_avg_vec)
    
    # Read the full data
    ramd_log_chunks, num_chunks = parser.make_log_file_chunks(ramd_log_file_list)
    for chunk_index, ramd_log_chunk in enumerate(ramd_log_chunks):
        trajectory_list = []
        trajectories_file_indices_list = []
        trajectories_starting_step_list = []
        for ramd_log_filename in ramd_log_chunk:
            trajectories, force_out_freq_, position_print_freq_, forceRAMD_, \
                time_step_, num_trajectories_, temperature_, max_dist_, \
                num_exits, trajectories_file_indices, trajectories_starting_step\
                    = parser.parse_ramd_log_file(ramd_log_filename)
            num_total_trajs += num_trajectories_
            num_total_exits += num_exits
            if force_out_freq is None:
                force_out_freq = force_out_freq_
                position_print_freq = position_print_freq_
                forceRAMD = forceRAMD_
                time_step = time_step_
                temperature = temperature_
                max_dist = max_dist_
            else:
                assert force_out_freq == force_out_freq_, \
                    "RAMD logs contain differing force_out_freq values."
                assert position_print_freq == position_print_freq_, \
                    "RAMD logs contain differing position_print_freq values."
                assert forceRAMD == forceRAMD_, \
                    "RAMD logs contain differing forceRAMD values."
                assert time_step == time_step_, \
                    "RAMD logs contain differing time_step values."
                assert temperature == temperature_, \
                    "RAMD logs contain differing temperature values."
                assert max_dist == max_dist_, \
                    "RAMD logs contain differing max_dist values."
                
            trajectory_list += trajectories
            trajectories_file_indices_list += trajectories_file_indices
            trajectories_starting_step_list += trajectories_starting_step
        
        beta = 1.0 / (temperature * unit.kelvin \
                      * samples.kB_kcal_per_mol_per_kelvin)
        unitless_beta = beta.value_in_unit(unit.mole / unit.kilocalorie)
        # Each trajectory represents a set of frames between force changes
        # Align the trajectories and convert to 1D, and xi values
        one_dim_trajs = parser.condense_trajectories(
            trajectory_list, max_dist, norm_avg_vec, norm_avgx, norm_avgy, 
            norm_avgz)
        
        #Construct a milestoning model based on trajectory fragments
        if milestones is None:
            milestones, min_location, max_location, starting_cv_val, \
                starting_index, h, situation \
                = milestoning.uniform_milestone_locations(
                    one_dim_trajs, num_milestones, max_dist)
    
            if situation == 1:
                # a forward exit (parallel to CV)
                bulk_milestones = [num_milestones - 1]
            elif situation == 2:
                # a backward exit (antiparallel to CV)
                bulk_milestones = [0]
            else:
                bulk_milestones = [0, num_milestones - 1]
                
            if debug_directory is not None:
                milestones_filename = os.path.join(debug_directory, 
                                                   "milestones.txt")
                np.savetxt(milestones_filename, np.array(milestones), fmt="%.3e")
            
            if verbose:
                np.set_printoptions(precision=3)
                print("milestones:", np.array(milestones), "h:", f"{h:.3f} (A)")
                print("starting_index:", starting_index)
            
            xi_bins = samples.make_xi_bins(
                num_angle_bins, num_milestones, milestones, situation, 
                debug_directory)
        
        xi_bin_trajs = samples.make_xi_bin_trajs(
            num_angle_bins, num_milestones, milestones, situation)
        
        for i, trajectory in enumerate(one_dim_trajs):
            if len(trajectory) == 0:
                continue
            traj_xi = trajectory[0][1]
            found_bin = False
            for j, xi_bin_traj in enumerate(xi_bin_trajs):
                if (traj_xi >= xi_bin_traj.lowest_boundary) \
                        and (traj_xi <= xi_bin_traj.highest_boundary):
                    xi_bin_traj.trajs.append(trajectory)
                    xi_bin_traj.traj_file_indices.append(
                        trajectories_file_indices_list[i])
                    xi_bin_traj.traj_starting_steps.append(
                        trajectories_starting_step_list[i])
                    found_bin = True
                    break
            
            if not found_bin:
                raise Exception(f"xi value {traj_xi} not placed in bin.")
        
        frame_time = time_step * position_print_freq
    
        for i, xi_bin_traj in enumerate(xi_bin_trajs):
            if verbose:
                print(f"num trajectories in bin {i}: {len(xi_bin_traj.trajs)}")
            if len(xi_bin_traj.trajs) == 0 and num_chunks == 1:
                print("All angle bins need trajectories. Either run more "\
                      "simulations or decrease the number of angle bins.")
                raise InsufficientTrajectoriesError
            
            if debug_directory is None:
                debug_filename = None
            else:
                debug_filename = os.path.join(
                    debug_directory, 
                    f"milestone_stats_{i}_chunk_{chunk_index}.log")
                
            count_matrix_chunk, time_vector_chunk, time_matrix_chunk \
                = milestoning.make_milestoning_model(
                    xi_bin_traj, milestones, frame_time, num_milestones,
                    remove_unbalanced_frame_counts=cut_out_drift, 
                    debug_filename=debug_filename, 
                    trajectory_stride=trajectory_stride)
            
            xi_bins[i].count_matrix += count_matrix_chunk
            xi_bins[i].time_vector += time_vector_chunk
            xi_bins[i].time_matrix += time_matrix_chunk
            
        if convergence:
            if debug_directory is None:
                debug_directory_for_conv = None
            else:
                debug_directory_for_conv_base = os.path.join(debug_directory, 
                                                             "converge")
                if not os.path.exists(debug_directory_for_conv_base):
                    os.mkdir(debug_directory_for_conv_base)
                debug_directory_for_conv = os.path.join(
                    debug_directory_for_conv_base, f"frame_{chunk_index}")
                if not os.path.exists(debug_directory_for_conv):
                    os.mkdir(debug_directory_for_conv)
            analyze_for_conv = analyze_traj_set(
                xi_bins, starting_index, num_milestones, milestones, h, 
                situation, bulk_milestones, beta, unitless_beta, forceRAMD, 
                CONVERGE_NUM_ERROR_SAMPLES, flatten_beyond_zero_count, 
                stride_error_samples, skip_error_samples, verbose=False, 
                debug_directory=debug_directory_for_conv)
            if debug_directory is not None:
                analyze_for_conv.create_plots(debug_directory_for_conv)
                make_xi_bin_plots(milestones, xi_bins, debug_directory_for_conv)
                
            converge_list.append(analyze_for_conv)
    
    print("log files read and milestoning models chosen")

    # Finished reading all log files
    if verbose:
        print("Number of trajectories extracted from log file(s):", 
              num_total_trajs)
        print("force_out_freq:", force_out_freq)
        print("position_print_freq:", position_print_freq)
        print("forceRAMD:", forceRAMD, "kcal/(mole * Angstrom)")
        print("time_step:", time_step, "ps")
        print("temperature:", temperature, "K")
        print("max_dist:", max_dist, "A")
        print("beta:", beta)
        
    print(f"{num_total_exits} exited successfully of "\
          f"{num_total_log_files} log files.")
    analysis = analyze_traj_set(
        xi_bins, starting_index, num_milestones, milestones, h, situation, 
        bulk_milestones, beta, unitless_beta, forceRAMD, num_error_samples, 
        flatten_beyond_zero_count, stride_error_samples, skip_error_samples, 
        verbose=verbose, debug_directory=debug_directory)
    if display_xi_bin_plots:
        make_xi_bin_plots(milestones, xi_bins, image_directory)
    print("analysis completed")
    return analysis, converge_list

def converge(converge_list, num_milestones, num_angle_bins, 
             image_directory=None, toy_mode=False):
    num_intervals = len(converge_list)
    num_log_files = num_intervals * CONV_PRINT_UPDATE_INTERVAL
    residence_times = np.zeros((num_intervals+1))
    residence_times_err = np.zeros((num_intervals+1))
    log_indices = np.zeros((num_intervals+1))
    for i, conv_analysis in enumerate(converge_list):
        interval_num = i * CONV_PRINT_UPDATE_INTERVAL
        print("Now analyzing convergence interval:", interval_num, "of", 
               num_log_files, "log files.")
        log_indices[i] = i
        try:
            residence_times[i] = conv_analysis.MFPT
            residence_times_err[i] = conv_analysis.MFPT_error
        except InsufficientTrajectoriesError:
            residence_times[i] = np.NAN
            continue
        
    if not np.any(np.isfinite(residence_times)) or np.all(residence_times == 0):
        # No entries, therefore don't make a plot
        return
    
    fig, ax = plt.subplots()
    
    ax.errorbar(log_indices, residence_times, yerr=residence_times_err,
            linestyle="-", marker="o", markersize=2, capsize=2, color="k", 
            ecolor="k")
    # get finite min and max values
    minval = 0.8*np.nanmin(np.ma.where(
        np.logical_and(np.isfinite(residence_times), residence_times != 0), 
        residence_times, np.nan)) + 1.0
    maxval = 1.2*np.nanmax(np.ma.where(
        np.logical_and(np.isfinite(residence_times), residence_times != 0), 
        residence_times, np.nan))
    plt.ylim((minval, maxval))
    plt.ylabel("Residence Time (ps)")
    plt.xlabel("Number of log files included")
    plt.title("Residence Time Convergence")
    plt.yscale("log", nonpositive="mask")
    plt.tight_layout()
    if image_directory is None:
        plt.show()
    else:
        plt.savefig(os.path.join(image_directory, "converge.png"))
    
    plt.close(fig)
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "ramd_log_files", metavar="RAMDLOGFILES", type=str, nargs="+",
        help="The RAMD log files to open and parse.")
    argparser.add_argument(
        "-n", "--num_milestones", dest="num_milestones", default=20,
        help="The number of milestones to use in the milestoning "\
        "calculation.", type=int)
    argparser.add_argument(
        "-a", "--num_angle_bins", dest="num_angle_bins", default=10,
        help="The number of bins to divide the force vectors into.", type=int)
    argparser.add_argument(
        "-t", "--toy_mode", dest="toy_mode", default=False,
        help="In toy mode, we assume that the main CV of the simulation is "\
        "the Z-axis of the zeroth particle. In contrast, the CV of a non-"
        "toy system is found using a more complex procedure.", 
        action="store_true")
    argparser.add_argument(
        "-N", "--num_error_samples", dest="num_error_samples", 
        default=1000, type=int, help="Specify the number of error samples" \
        " to generate for estimating error/uncertainty of computed "\
        "values. Default: 1000")
    argparser.add_argument(
        "-S", "--stride_error_samples", dest="stride_error_samples", 
        default=None, type=int, help="Specify the number of strides between" \
        "saved error samples. An argument of None automatically assigns the "\
        "quantity at the number of milestones in the model squared. Default: "\
        "None")
    argparser.add_argument(
        "-K", "--skip_error_samples", dest="skip_error_samples", 
        default=None, type=int, help="Specify the number of error samples. "\
        "An argument of None automatically assigns the quantity at ten times "\
        "the number of milestones in the model squared. Default: None.")
    argparser.add_argument(
        "-c", "--cut_out_drift", dest="cut_out_drift", default=False,
        help="Exact milestoning theory requires truly equilibrium "\
        "trajectories. RAMD simulations aren't that. This will attempt to cut "\
        "the most non-equilibrium situations where there is drift between "\
        "starting and ending locations.", action="store_true")
    argparser.add_argument(
        "-d", "--image_directory", dest="image_directory", 
        default=None, type=str,
        help="Define the directory where all plots and images will be saved.")
    argparser.add_argument(
        "-v", "--verbose", dest="verbose", default=False,
        help="Verbose mode will print out much more information about the "\
        "milestoning model, and other information.", action="store_true")
    argparser.add_argument(
        "-x", "--display_xi_bin_plots", dest="display_xi_bin_plots", 
        default=False, help="Toggle the display of energy plots for each xi "\
        "bin, both slanted and unslanted.", action="store_true")
    argparser.add_argument(
        "-C", "--convergence", dest="convergence", default=False,
        help="Perform a convergence analysis and generate a convergence plot", 
        action="store_true")
    argparser.add_argument(
        "-f", "--dont_flatten_beyond_zero_count", 
        dest="dont_flatten_beyond_zero_count", default=True, 
        help="Statistics are difficult to obtain for Theta RAMD milestoning "\
        "results in high-energy regions. Once a zero-count entry is "\
        "encountered, the program will, by default, assume that this is near "\
        "the highest point of the barrier, and will flatten the statistics "\
        "beyond that point, to counteract stochasticity, that would otherwise "\
        "generate strange results if very little sampling is performed. "\
        "If this flag is called, that function will be deactivated", 
        action="store_false")
    argparser.add_argument(
        "-D", "--debug_directory", dest="debug_directory", default=None,
        help="Supply a directory to write debug files - milestoning "\
        "statistics, vectors, and matrices. If None is provided, then the "\
        "defaults won't be written. Default: None.")
    argparser.add_argument(
        "-T", "--trajectory_stride", dest="trajectory_stride", 
        default=1, type=int, help="Provide the trajectory stride for RAMD "\
            "simulations for debug purposes. Doesn't do anything unless "\
            "-D is provided to indicate debug mode.")
    
    args = argparser.parse_args()
    args = vars(args)
    ramd_log_files = args["ramd_log_files"]
    num_milestones = args["num_milestones"]
    num_angle_bins = args["num_angle_bins"]
    toy_mode = args["toy_mode"]
    num_error_samples = args["num_error_samples"]
    stride_error_samples = args["stride_error_samples"]
    skip_error_samples = args["skip_error_samples"]
    cut_out_drift = args["cut_out_drift"]
    image_directory = args["image_directory"]
    verbose = args["verbose"]
    display_xi_bin_plots = args["display_xi_bin_plots"]
    convergence = args["convergence"]
    flatten_beyond_zero_count = args["dont_flatten_beyond_zero_count"]
    debug_directory = args["debug_directory"]
    trajectory_stride = args["trajectory_stride"]
    
    analysis, converge_list = theta_ramd(
        ramd_log_files, num_milestones, num_angle_bins, toy_mode=toy_mode, 
        num_error_samples=num_error_samples, 
        stride_error_samples=stride_error_samples, 
        skip_error_samples=skip_error_samples, cut_out_drift=cut_out_drift, 
        image_directory=image_directory, verbose=verbose, 
        display_xi_bin_plots=display_xi_bin_plots, 
        flatten_beyond_zero_count=flatten_beyond_zero_count,
        debug_directory=debug_directory, trajectory_stride=trajectory_stride,
        convergence=convergence)
    
    analysis.print_results()
    if convergence:
        converge(converge_list, num_milestones=num_milestones, 
                 num_angle_bins=num_angle_bins, image_directory=image_directory,
                 toy_mode=toy_mode)
        
    analysis.create_plots(image_directory)
