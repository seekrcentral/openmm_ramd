"""
maximova.py

Implements the RAMD analysis method presented by Maximova et al. in the 
publication:

Maximova E, Postnikov EB, Lavrova AI, Farafonov V, Nerukh D. Protein-Ligand 
Dissociation Rate Constant from All-Atom Simulation. J Phys Chem Lett. 2021 
Nov 4;12(43):10631-10636. doi: 10.1021/acs.jpclett.1c02952. Epub 2021 Oct 27. 
PMID: 34704768.

Which implements a Kramer's rate theory-based fitting and extrapolation of
residence times from RAMD simulations of different force constants in order
to estimate the residence time of a ligand-receptor complex in the absence
of any RAMD force.
"""
import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import openmm_ramd.analyze.modules.parser as parser
import openmm_ramd.analyze.modules.samples as samples

NUM_MONTE_CARLO_SAMPLES = 100
NUM_ASSOC_PROP_STEPS = 1000
FS_PER_PS = 1000

def plot_fitting_extrapol(force_list, log_log_time_list, polynomial, 
                          intercept_avg, intercept_std, image_directory=None):
    """
    
    """
    data = pd.DataFrame({"x": force_list,
                         "y": log_log_time_list})
    fig, ax = plt.subplots()
    ax.set_xlim(left=-0.01, right=np.max(force_list)+0.01)
    
    ax = sns.regplot(data=data, x="x", y="y", truncate=False, ci=66.67, ax=ax)
    ax.set(xlabel="$f$, kcal/(mol $\\cdot$ A)", ylabel="ln(ln($\\tau$ / 1fs))", 
           title="RAMD Force Effect on Residence Time(s)")
    ax.plot([0.0], [intercept_avg], marker="*")
    
    if image_directory is None:
        plt.show()
    else:
        plt.savefig(os.path.join(image_directory, "maximova_extrapolation.png"))
    
    plt.close(fig)
    return

def plot_association_proportion(force_const_dict_times, image_directory=None):
    """
    
    """
    for force_const, end_times in force_const_dict_times.items():
        max_time = np.max(end_times)
        N_0 = len(end_times)
        x = np.linspace(0.0, max_time, NUM_ASSOC_PROP_STEPS)
        y = np.zeros(NUM_ASSOC_PROP_STEPS)
        for i in range(NUM_ASSOC_PROP_STEPS):
            N_t = len([j for j in end_times if j >= x[i]])
            y[i] = np.log(N_t / N_0)
            
        plt.scatter(x, y, marker="o", facecolors="none", edgecolors="r")
        plt.title(f"Association Probability at Force {force_const} "\
                  "kcal/(mol $\\cdot$ A)")
        plt.ylabel("ln($N(t)$/$N(0)$)")
        plt.xlabel("$t$ ps")
        plt.tight_layout()
        if image_directory is None:
            plt.show()
        else:
            plt.savefig(os.path.join(
                image_directory, "maximova_association_probabilities.png"))
        
        plt.close()

def fit_and_extrapolate(force_const_dict_times, image_directory=None, 
                        make_plot=True, silent=False):
    """
    
    """
    # First, compute avg of all
    force_list = []
    log_log_time_list = []
    for force_const, times in force_const_dict_times.items():
        for time in times:
            force_list.append(force_const)
            log_log_time = np.log(np.log(time))
            log_log_time_list.append(log_log_time)
    
    if len(force_list) > 2:
        coeffs, cov_mat = np.polyfit(force_list, log_log_time_list, 1, 
                                     cov="unscaled")
        intercept_std = cov_mat[1,1]
    else:
        coeffs = np.polyfit(force_list, log_log_time_list, 1, cov=False)
        intercept_std = None
    
    polynomial = np.poly1d(coeffs)
    intercept_avg = coeffs[1]
    
    time_sampled_values = []
    for _ in range(NUM_MONTE_CARLO_SAMPLES):
        log_log_time_sample = np.random.normal(
            loc=intercept_avg, scale=intercept_std)
        time_sampled_values.append(np.exp(np.exp(log_log_time_sample)) \
                                   / FS_PER_PS)
    
    median_time = np.median(time_sampled_values)
    time_0 = np.exp(np.exp(intercept_avg)) / FS_PER_PS
    
    times_minus_median = []
    for time in time_sampled_values:
        times_minus_median.append(abs(time - median_time))
        
    median_diff = np.median(times_minus_median)
    
    if not silent:
        print("Printing values extrapolated to zero force:")
        print("ln(ln(tau_0/fs)):", samples.pretty_string_value_error(
            intercept_avg, intercept_std))
        print("tau_0:", samples.pretty_string_value_error(
            time_0, median_diff), "ps")
    
    if make_plot:
        plot_fitting_extrapol(
            force_list, log_log_time_list, polynomial, intercept_avg, intercept_std,
            image_directory)
    
    return time_0, median_diff

def maximova(ramd_log_files, include_all_times=False, image_directory=None,
             make_plot=True, silent=False):
    force_const_dict_steps = defaultdict(list)
    time_step = None
    if not silent:
        print("Reading log files...")
    for ramd_log_filename in ramd_log_files:
        end_steps, forceRAMD_, time_step_ \
            = parser.parse_ramd_log_file_quick(
                ramd_log_filename, include_all_times)
        
        force_const_dict_steps[forceRAMD_] += end_steps
        
        if time_step is None:
            time_step = time_step_
        else:
            assert time_step == time_step_, "Different timesteps found "\
                "in RAMD input files. All files must have the same timesteps"
        
    force_const_dict_times = defaultdict(list)
    for force_const, end_steps in force_const_dict_steps.items():
        for end_step in end_steps:
            if end_step is not None:
                force_const_dict_times[force_const].append(
                    end_step * time_step * FS_PER_PS)
    
    if not silent:
        for force_const, end_times in force_const_dict_times.items():
            print(f"Found {len(end_times)} trajectories for force constant "\
                  f"{force_const} kcal/(mol * A).")
        
    assert len(force_const_dict_steps) > 1, "Simulations with more than one "\
        "force constant needed to perform fitting and extrapolation."
    
    if not silent:
        print("Fitting and extrapolating data points...")
    tau_0, tau_0_error = fit_and_extrapolate(force_const_dict_times,
                                             image_directory, make_plot, silent)
    
    return tau_0, tau_0_error, force_const_dict_times
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "ramd_log_files", metavar="RAMDLOGFILES", type=str, nargs="+",
        help="The RAMD log files to open and parse.")
    argparser.add_argument(
        "-i", "--include_all_times", dest="include_all_times", default=False, 
        help="By default, only the trajectories that end successfully will "\
        "be considered by the parser. If set to True, this flag will cause "\
        "the times of even the unfinished trajectories to be counted. "\
        "Default: False.", action="store_true")
    argparser.add_argument(
        "-p", "--make_probability_plots", dest="make_probability_plots", 
        default=False, 
        help="Generate plots that depict association probability over time. "\
        "Default: False.", action="store_true")
    argparser.add_argument(
        "-d", "--image_directory", dest="image_directory", 
        default=None, type=str,
        help="Define the directory where all plots and images will be saved.")
    
    args = argparser.parse_args()
    args = vars(args)
    ramd_log_files = args["ramd_log_files"]
    include_all_times = args["include_all_times"]
    make_probability_plots = args["make_probability_plots"]
    image_directory = args["image_directory"]
    
    tau_0, tau_0_error, force_const_dict_times = maximova(
        ramd_log_files, include_all_times, image_directory)
    if make_probability_plots:
        plot_association_proportion(force_const_dict_times, image_directory)