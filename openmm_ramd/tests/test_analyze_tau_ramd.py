"""
test_analyze_tau_ramd.py

Tests for the analyze.tau_ramd module.
"""

import os

import openmm_ramd.analyze.tau_ramd as tau_ramd

TEST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

test_small_log_end = os.path.join(TEST_DIRECTORY, "data/small_with_end.txt")
test_large_log = os.path.join(TEST_DIRECTORY, "data/large_linear.txt")

def test_tau_ramd():
    ramd_log_files = [test_small_log_end, test_small_log_end]
    tau_median, tau_mean, t_r, SD_r, num_exits, num_sims = tau_ramd.tau_ramd(
            ramd_log_files, include_all_times=False)
    num_steps = 5612050
    timestep = 0.002
    assert tau_median == num_steps * timestep
    assert tau_mean == num_steps * timestep
    assert t_r == num_steps * timestep
    assert SD_r == 0
    assert num_exits == 2
    assert num_sims == 2

    ramd_log_files = [test_large_log]
    tau_median, tau_mean, t_r, SD_r, num_exits, num_sims = tau_ramd.tau_ramd(
            ramd_log_files, include_all_times=True)
    assert num_sims == 101
    return

