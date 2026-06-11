"""
Implements the Tau-RAMD method as described in the publication:

Estimation of Drug-Target Residence Times by τ-Random Acceleration Molecular 
Dynamics Simulations. Daria B. Kokh, Marta Amaral, Joerg Bomke, 
Ulrich Grädler, Djordje Musil, Hans-Peter Buchstaller, Matthias K. Dreyer, 
Matthias Frech, Maryse Lowinski, Francois Vallee, Marc Bianciotto, Alexey Rak, 
and Rebecca C. Wade. Journal of Chemical Theory and Computation 2018 14 (7), 
3859-3869 DOI: 10.1021/acs.jctc.8b00230 

This script groups RAMD log files by starting replica, where ramd_0_* belongs
to replica 1, ramd_1_* belongs to replica 2, etc. It reports the interim
Tau-RAMD values for each replica, then averages over replicas to compute
tau_comp and SD_comp as described in the paper above.
"""

import argparse
import os
import re
import statistics
import random

import openmm_ramd.analyze.base as analyze_base

REPLICA_LOG_PATTERN = re.compile(r"ramd_(\d+)_.+\.log$")

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
    # NOT the same as tau_comp, which involves the mean of bootstrap times across replicas
    tau_median = statistics.median(completed_times)
    tau_mean = statistics.mean(completed_times)
    
    # interim residence time
    t_r = statistics.mean(bootstrap_times)
    
    # interim standard deviation of residence time distribution
    SD_r = statistics.stdev(bootstrap_times)
    
    return tau_median, tau_mean, t_r, SD_r

def tau_ramd(ramd_log_files, include_all_times=False):
    forceRAMD = None
    time_step = None
    all_end_steps = []
    for ramd_log_filename in ramd_log_files:
        end_steps, forceRAMD_, time_step_ \
            = analyze_base.parse_ramd_log_file_quick(
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
    
    tau_median, tau_mean, t_r, SD_r = bootstrap(completed_times)
    num_exits = len(completed_times)
    return tau_median, tau_mean, t_r, SD_r, num_exits, num_sims

def get_replica_index(ramd_log_file):
    basename = os.path.basename(ramd_log_file)
    match = REPLICA_LOG_PATTERN.match(basename)
    if match is None:
        return None
    return int(match.group(1))

def group_log_files_by_replica(ramd_log_files):
    replica_log_files = {}
    unmatched_log_files = []
    for ramd_log_file in ramd_log_files:
        replica_index = get_replica_index(ramd_log_file)
        if replica_index is None:
            unmatched_log_files.append(ramd_log_file)
        else:
            replica_log_files.setdefault(replica_index, []).append(ramd_log_file)
    
    assert len(unmatched_log_files) == 0, \
        "All RAMD log files must match the pattern ramd_<replica>_*.log. "\
        f"Unmatched files: {unmatched_log_files}"
    return [
        (replica_index, sorted(replica_log_files[replica_index]))
        for replica_index in sorted(replica_log_files)
    ]

def tau_ramd_replicas(ramd_log_files, include_all_times=False):
    replica_results = []
    for replica_index, replica_log_files in group_log_files_by_replica(ramd_log_files):
        tau_median, tau_mean, t_r, SD_r, num_exits, num_sims = tau_ramd(
            replica_log_files, include_all_times)
        replica_results.append({
            "replica_index": replica_index,
            "ramd_log_files": replica_log_files,
            "tau_median": tau_median,
            "tau_mean": tau_mean,
            "t_r": t_r,
            "SD_r": SD_r,
            "num_exits": num_exits,
            "num_sims": num_sims,
        })
    
    assert len(replica_results) > 0, "No RAMD replica log files were provided."
    t_r_values = [replica_result["t_r"] for replica_result in replica_results]
    SD_r_values = [replica_result["SD_r"] for replica_result in replica_results]
    tau_comp = statistics.mean(t_r_values)
    avg_SD_r = statistics.mean(SD_r_values)
    if len(t_r_values) > 1:
        SD_replicas = statistics.stdev(t_r_values)
    else:
        SD_replicas = 0.0
    SD_comp = max(avg_SD_r, SD_replicas)
    return replica_results, tau_comp, SD_comp, avg_SD_r, SD_replicas

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
    
    replica_results, tau_comp, SD_comp, avg_SD_r, SD_replicas \
        = tau_ramd_replicas(ramd_log_files, include_all_times)
    
    for replica_result in replica_results:
        replica_number = replica_result["replica_index"] + 1
        num_exits = replica_result["num_exits"]
        num_sims = replica_result["num_sims"]
        print(f"Replica {replica_number}: {num_exits} of {num_sims} trajectories exited successfully.")
        print("Tau-RAMD results for this replica:")
        print("tau_median (residence time):", replica_result["tau_median"], "ps")
        print("tau mean (residence time):", replica_result["tau_mean"], "ps")
        print("t_r (median interim residence time):", replica_result["t_r"], "ps")
        print("SD_r (interim residence time standard deviation):", replica_result["SD_r"], "ps")
    
    print("Tau-RAMD results averaged over replicas:")
    print("tau_comp (average t_r over replicas):", tau_comp, "ps")
    print("SD_comp (computed residence time uncertainty):", SD_comp, "ps")
    print("average SD_r over replicas:", avg_SD_r, "ps")
    print("SD_replicas (standard deviation of t_r over replicas):", SD_replicas, "ps")
