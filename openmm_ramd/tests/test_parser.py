"""
Test openmm_ramd parser.py functions.
"""

import numpy as np

import openmm_ramd.analyze.parser as parser

def test_parse_ramd_log_file(small_linear_logfile_persistent):
    [trajectories, forceOutFreq, forceRAMD, timeStep, num_simulations,
     temperature] = small_linear_logfile_persistent
    assert forceOutFreq == 50
    assert forceRAMD == 0.1
    assert timeStep == 0.002
    assert len(trajectories) == 2
    assert num_simulations == 1
    assert temperature == 300.0
    assert np.isclose(trajectories[0][0][0], -0.11735814)
    assert np.isclose(trajectories[0][0][1], 0.59439099)
    assert np.isclose(trajectories[0][0][2], 0.07284915)
    assert np.isclose(trajectories[0][0][3], 0.46881796)
    assert np.isclose(trajectories[0][0][4], 0.82472121)
    assert np.isclose(trajectories[0][0][5], -0.31629834)
    
    assert np.isclose(trajectories[1][0][0], 2.16828823)
    assert np.isclose(trajectories[1][0][1], 16.58750725)
    assert np.isclose(trajectories[1][0][2], 0.47683921)
    assert np.isclose(trajectories[1][0][3], 0.20150257)
    assert np.isclose(trajectories[1][0][4], 0.27089315)
    assert np.isclose(trajectories[1][0][5], 0.94128296)
    
    assert np.isclose(trajectories[1][-1][0], 7.21598816)
    assert np.isclose(trajectories[1][-1][1], 24.0497818)
    assert np.isclose(trajectories[1][-1][2], 10.65902901)
    assert np.isclose(trajectories[1][-1][3], 0.20150257)
    assert np.isclose(trajectories[1][-1][4], 0.27089315)
    assert np.isclose(trajectories[1][-1][5], 0.94128296)
    return

# TODO: will need the same tests for log files with receptor motion

def test_parse_ramd_log_file_large(large_linear_logfile_persistent):
    [trajectories, forceOutFreq, forceRAMD, timeStep, num_simulations,
     temperature] = large_linear_logfile_persistent
    assert forceOutFreq == 50
    assert forceRAMD == 0.1
    assert timeStep == 0.002
    assert num_simulations == 101
    assert len(trajectories) == 242
    return

def test_condense_trajectories(small_linear_logfile_persistent):
    [trajectories, forceOutFreq, forceRAMD, timeStep, num_simulations,
     temperature] = small_linear_logfile_persistent
    one_dim_trajs = parser.condense_trajectories(trajectories, onlyZ=True)
    for trajectory, one_dim_traj in zip(trajectories, one_dim_trajs):
        for frame, one_dim_frame in zip(trajectory, one_dim_traj):
            assert frame[2] == one_dim_frame[0]
            assert frame[5] == one_dim_frame[1]
            
    # TODO: make test (perhaps) for onlyZ=False