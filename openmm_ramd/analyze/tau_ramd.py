"""
Implements the Tau-RAMD method as described in the publication:

Estimation of Drug-Target Residence Times by τ-Random Acceleration Molecular 
Dynamics Simulations. Daria B. Kokh, Marta Amaral, Joerg Bomke, 
Ulrich Grädler, Djordje Musil, Hans-Peter Buchstaller, Matthias K. Dreyer, 
Matthias Frech, Maryse Lowinski, Francois Vallee, Marc Bianciotto, Alexey Rak, 
and Rebecca C. Wade. Journal of Chemical Theory and Computation 2018 14 (7), 
3859-3869 DOI: 10.1021/acs.jctc.8b00230 

Specifically, this finds the tau_

This script ought to be run on the results of one replica at a time, and then
the results manually averaged over all replicas in order to recreate the
method in the paper above.
"""

import argparse
import statistics
import random

import openmm_ramd.analyze.modules.parser as parser

def bootstrap(completed_times, fraction=0.8, num_instances=200):
    """
    Implementing the bootstrapping procedure for Tau RAMD - select 'fraction'
    0.8 (or other fraction) of the data, compute the median time. This
    is then repeated 'num_instances' (200 by default) to get a distribution
    of times in order to compute mean and std. dev.
    """
    fraction_count = int(fraction * len(completed_times))
    bootstrap_times = []
    for _ in range(num_instances):
        bootstrap_sequence = random.sample(completed_times, fraction_count)
        # The time taken for 50% of the trajectories to escape
        bootstrap_time = statistics.median(bootstrap_sequence)
        bootstrap_times.append(bootstrap_time)
    
    # The residence time
    # The time taken for 50% of the trajectories to escape
    tau_comp = statistics.median(completed_times)
    tau_mean = statistics.mean(completed_times)
    
    # interim residence time
    t_r = statistics.mean(bootstrap_times)
    
    # interim standard deviation of residence time distribution
    SD_r = statistics.stdev(bootstrap_times)
    
    return tau_comp, tau_mean, t_r, SD_r

def tau_ramd(ramd_log_files, include_all_times=False):
    forceRAMD = None
    time_step = None
    all_end_steps = []
    for ramd_log_filename in ramd_log_files:
        end_steps, forceRAMD_, time_step_ \
            = parser.parse_ramd_log_file_quick(
                ramd_log_filename, include_all_times)
        
        if forceRAMD is None:
            forceRAMD = forceRAMD_
            time_step = time_step_
        else:
            assert forceRAMD == forceRAMD_, "Different force constants "\
                "found in RAMD input files. All files must have the same forces"
            assert time_step == time_step_, "Different timesteps found "\
                "in RAMD input files. All files must have the same timesteps."
        
        all_end_steps += end_steps
        
    completed_times = []
    num_sims = 0
    for end_step in all_end_steps:
        num_sims += 1
        if end_step is not None:
            completed_times.append(end_step * time_step)
    
    assert len(completed_times) > 0, \
        "No escape events in log files. Use '-i' argument?"
    assert len(completed_times) > 1, \
        "More than one completed trajectory needed"
    
    tau_comp, tau_mean, t_r, SD_r = bootstrap(completed_times)
    num_exits = len(completed_times)
    return tau_comp, tau_mean, t_r, SD_r, num_exits, num_sims

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
    
    args = argparser.parse_args()
    args = vars(args)
    ramd_log_files = args["ramd_log_files"]
    include_all_times = args["include_all_times"]
    
    tau_comp, tau_mean, t_r, SD_r, num_exits, num_sims \
        = tau_ramd(ramd_log_files, include_all_times)
    
    print(f"{num_exits} of {num_sims} trajectories exited successfully.")
    print("Tau-RAMD results for this set of log files:")
    print("tau_comp (residence time):", tau_comp, "ps")
    print("tau mean (residence time):", tau_mean, "ps")
    print("t_r (median interim residence time):", t_r, "ps")
    print("SD_r (interim residence time standard deviation):", SD_r, "ps")