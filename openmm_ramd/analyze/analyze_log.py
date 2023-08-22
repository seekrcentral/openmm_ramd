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

import openmm_ramd.analyze.parser as parser
import openmm_ramd.analyze.samples as samples
import openmm_ramd.analyze.milestoning as milestoning

def analyze_log(ramd_log_file_list, num_milestones=10, num_xi_bins=5, 
                verbose=True, toy_mode=False, num_error_samples=0, 
                stride_error_samples=None, skip_error_samples=None, 
                cut_out_drift=False):
    if verbose:
        print(f"Using {num_milestones} milestones.")
        print(f"Using {num_xi_bins} angle bins.")
    
    trajectory_list = []
    force_out_freq = None
    forceRAMD = None
    time_step = None
    temperature = None
    max_dist = None
    if verbose:
        print("Now loading log files...")
        
    for ramd_log_filename in ramd_log_file_list:
        trajectories, force_out_freq_, forceRAMD_, time_step_, \
            num_trajectories_, temperature_, max_dist_ \
            = parser.parse_ramd_log_file(
                ramd_log_filename)
        if force_out_freq is None:
            force_out_freq = force_out_freq_
            forceRAMD = forceRAMD_
            time_step = time_step_
            temperature = temperature_
            max_dist = max_dist_
        else:
            assert force_out_freq == force_out_freq_, \
                "RAMD logs contain differing force_out_freq values."
            assert forceRAMD == forceRAMD_, \
                "RAMD logs contain differing forceRAMD values."
            assert time_step == time_step_, \
                "RAMD logs contain differing time_step values."
            assert temperature == temperature_, \
                "RAMD logs contain differing temperature values."
            assert max_dist == max_dist_, \
                "RAMD logs contain differing max_dist values."
            
        trajectory_list += trajectories
    
    beta = 1.0 / (temperature * unit.kelvin \
                  * samples.kB_kcal_per_mol_per_kelvin)
    unitless_beta = beta.value_in_unit(unit.mole / unit.kilocalorie)
    if verbose:
        print("Number of trajectories extracted from log file(s):", 
              len(trajectory_list))
        print("force_out_freq:", force_out_freq)
        print("forceRAMD:", forceRAMD, "kcal/(mole * Angstrom)")
        print("time_step:", time_step, "ps")
        print("temperature:", temperature, "K")
        print("max_dist:", max_dist, "A")
        print("beta:", beta)
    
    # Each trajectory represents a set of frames between force changes
    # Align the trajectories and convert to 1D, and xi values
    one_dim_trajs = parser.condense_trajectories(
        trajectory_list, onlyZ=toy_mode)
        
    #Construct a milestoning model based on trajectory fragments
    milestones, min_location, max_location, starting_cv_val, starting_index, h \
        = milestoning.uniform_milestone_locations(one_dim_trajs, num_milestones,
                                                  max_dist)
    bulk_milestone = num_milestones - 1
    if verbose:
        np.set_printoptions(precision=3)
        print("milestones:", np.array(milestones), "h:", f"{h:.3f}")
        print("starting_index:", starting_index)
        
    xi_bins = samples.make_xi_bins(num_xi_bins, num_milestones, milestones)
    for i, trajectory in enumerate(one_dim_trajs):
        traj_xi = trajectory[0][1]
        found_bin = False
        for j, xi_bin in enumerate(xi_bins):
            if (traj_xi >= xi_bin.lowest_boundary) \
                    and (traj_xi <= xi_bin.highest_boundary):
                xi_bin.trajs.append(trajectory)
                found_bin = True
                break
        
        if not found_bin:
            raise Exception(f"xi value {traj_xi} not placed in bin.")
    
    frame_time = time_step * force_out_freq
    forceRAMD_in_nm_unitless = forceRAMD * samples.A_per_nm
    forceRAMD_in_nm = forceRAMD_in_nm_unitless * unit.kilocalories_per_mole
    forceRAMD_in_A = forceRAMD * unit.kilocalories_per_mole
    
    for i, xi_bin in enumerate(xi_bins):
        xi = xi_bin.midpoint
        if verbose:
            print(f"num trajectories in bin {i}: {len(xi_bin.trajs)}")
        assert len(xi_bin.trajs) > 0, "All angle bins need trajectories. "\
            "Either run more simulations or decrease the number of angle bins."
        count_matrix, time_vector, time_vector_up, time_vector_down, \
            avg_time_vector, avg_time_vector_up, avg_time_vector_down \
            = milestoning.make_milestoning_model(
                xi_bin.trajs, milestones, frame_time, num_milestones,
                remove_unbalanced_frame_counts=cut_out_drift)
        xi_bin.assign_main_data_sample(
            count_matrix, time_vector, avg_time_vector)
        if num_error_samples > 0:
            xi_bin.sample_error_matrices(
                num_error_samples, stride_error_samples, skip_error_samples)
        xi_bin.make_profiles(starting_index, wrap_sink=False, 
                                     unitless_beta=unitless_beta)
        xi_bin.unslant_data_samples(forceRAMD_in_nm_unitless, beta)
        xi_bin.counts = milestoning.compute_counts(count_matrix, num_milestones)
        xi_bin.make_profiles(starting_index, wrap_sink=False, 
                             unitless_beta=unitless_beta, 
                             data_sample=xi_bin.main_data_sample_unslanted)
    
    analysis = samples.Analysis(num_milestones, milestones, h)
    analysis.avg_unslanted_xi_bins(xi_bins)
    analysis.make_profiles(starting_index, wrap_sink=False, 
                           unitless_beta=unitless_beta)
    analysis.make_times(starting_index, bulk_milestone)
    return analysis



# TODO: add Torque balancing in the RAMD.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "ramd_log_files", metavar="RAMDLOGFILES", type=str, nargs="+",
        help="The RAMD log files to open and parse.")
    argparser.add_argument(
        "-n", "--num_milestones", dest="num_milestones", default=10,
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
        default=100, type=int, help="Specify the number of error samples" \
        " to generate for estimating error/uncertainty of computed "\
        "values. Default: 100")
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
    
    analysis = analyze_log(
        ramd_log_files, num_milestones, num_angle_bins, toy_mode=toy_mode, 
        num_error_samples=num_error_samples, 
        stride_error_samples=stride_error_samples, 
        skip_error_samples=skip_error_samples, cut_out_drift=cut_out_drift)
    
    analysis.print_results()
    #analysis.create_plots()