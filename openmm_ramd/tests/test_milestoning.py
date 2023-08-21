"""
Test openmm_ramd milestoning.py functions.
"""

import numpy as np

import utils
import openmm_ramd.analyze.parser as parser
import openmm_ramd.analyze.milestoning as milestoning

def test_uniform_milestone_locations(small_linear_logfile_persistent):
    num_milestones = 10
    [trajectories, forceOutFreq, forceRAMD, timeStep, num_simulations,
     temperature, maxDist] = small_linear_logfile_persistent
    one_dim_trajs = parser.condense_trajectories(trajectories, onlyZ=True)
    milestones, min_location, max_location, starting_cv_val, starting_index, h \
        = milestoning.uniform_milestone_locations(
            one_dim_trajs, num_milestones, maxDist)
    cv_vals = []
    for one_dim_traj in one_dim_trajs:
        for one_dim_frame in one_dim_traj:
            cv_val = one_dim_frame[0]
            cv_vals.append(cv_val)
    assert len(milestones) == num_milestones
    assert min_location == min(cv_vals)
    assert min_location < min(milestones)
    assert max_location == max(cv_vals)
    assert max_location > max(milestones)
    assert starting_cv_val == cv_vals[0]
    return

def test_make_milestoning_model_simple():
    milestones = [2.0, 4.0, 6.0]
    frame_time = 50.0
    num_milestones = 3
    xi_bin_trajs = [
        [[3.0, 0.0], [1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 0.0],
         [3.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0], [5.0, 0.0],
         [3.0, 0.0], [5.0, 0.0], [3.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [3.0, 0.0], [3.0, 0.0], [5.0, 0.0], [3.0, 0.0], [5.0, 0.0],
         [3.0, 0.0], [3.0, 0.0], [1.0, 0.0]]
    ]
    
    count_matrix, time_vector, time_vector_up, time_vector_down, \
        avg_time_vector, avg_time_vector_up, avg_time_vector_down, \
            = milestoning.make_milestoning_model(
            xi_bin_trajs, milestones, frame_time, num_milestones)
    true_count_matrix = np.array(
        [[0, 3, 0],
         [3, 0, 1],
         [0, 1, 0]])
    assert np.isclose(count_matrix, true_count_matrix).all()
    true_time_vector = np.array([
        [450.0],
        [550.0],
        [100.0]])
    #true_time_vector = np.array([
    #    [50.0],
    #    [150.0],
    #    [0.0]])
    assert np.isclose(time_vector, true_time_vector).all()
    assert np.isclose(time_vector_up + time_vector_down, true_time_vector).all()
    #true_transition_matrix = np.array(
    #    [[0, 0.75, 0],
    #     [1.0, 0, 1.0],
    #     [0, 0.25, 0]])
    #assert np.isclose(transition_matrix, true_transition_matrix).all()
    
    # TODO: make more tests for avg_time_vector_up/down
    
    true_rate_matrix = np.array(
        [[-3.0/550.0, 3.0/550.0,         0],
         [ 3.0/550.0,-4.0/550.0, 1.0/100.0],
         [ 0,         1.0/550.0,-1.0/100.0]])
    #assert np.isclose(rate_matrix, true_rate_matrix).all()
    return

def test_compute_stationary_probabilities_linear():
    num_milestones = 11
    start_milestone = 0
    h = 1.0
    D = 1.0
    t = h**2/(2.0*D)
    transition_matrix = np.zeros((num_milestones, num_milestones))
    time_vector = np.zeros((num_milestones, 1))
    transition_matrix[1, 0] = 1.0
    transition_matrix[num_milestones-2, num_milestones-1] = 1.0
    for i in range(1, num_milestones-1):
        transition_matrix[i-1, i] = 0.5
        transition_matrix[i+1, i] = 0.5
    
    for i in range(num_milestones):
        time_vector[i, 0] = t
    
    stationary_probs = milestoning.compute_stationary_probabilities(
        transition_matrix, time_vector, num_milestones, start_index=0)
    return
    
def test_compute_diffusion_coefficients():
    num_milestones = 11
    start_milestone = 0
    h = 1.0
    D = 1.0
    t = h**2/(2.0*D)
    transition_matrix = np.zeros((num_milestones, num_milestones))
    rate_matrix = np.zeros((num_milestones, num_milestones))
    time_vector = np.zeros((num_milestones, 1))
    transition_matrix[1, 0] = 1.0
    transition_matrix[num_milestones-2, num_milestones-1] = 1.0
    for i in range(1, num_milestones-1):
        transition_matrix[i-1, i] = 0.5
        transition_matrix[i+1, i] = 0.5
    
    for i in range(num_milestones):
        time_vector[i, 0] = t
    
    for i in range(num_milestones):
        for j in range(num_milestones):
            rate_matrix[i, j] = transition_matrix[i, j] / time_vector[j, 0]
    
    diffusion_coeffs = milestoning.compute_diffusion_coefficients(
        rate_matrix, num_milestones, h)
    
def test_compute_counts():
    num_milestones = 11
    count_matrix = np.zeros((num_milestones, num_milestones))
    count_matrix[1, 0] = 125
    count_matrix[num_milestones-2, num_milestones-1] = 30
    for i in range(1, num_milestones-1):
        count_matrix[i-1, i] = 55
        count_matrix[i+1, i] = 55
    counts = milestoning.compute_counts(count_matrix, num_milestones)