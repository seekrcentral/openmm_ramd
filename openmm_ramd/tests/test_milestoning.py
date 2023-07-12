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
    milestones, min_location, max_location, starting_cv_val, starting_index \
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
        transition_matrix, avg_time_vector, avg_time_vector_up, \
        avg_time_vector_down, rate_matrix \
            = milestoning.make_milestoning_model(
            xi_bin_trajs, milestones, frame_time, num_milestones)
    true_count_matrix = np.array(
        [[0, 3, 0],
         [3, 0, 1],
         [0, 1, 0]])
    assert np.isclose(count_matrix, true_count_matrix).all()
    true_time_vector = np.array([
        [550.0],
        [550.0],
        [100.0]])
    #true_time_vector = np.array([
    #    [50.0],
    #    [150.0],
    #    [0.0]])
    assert np.isclose(time_vector, true_time_vector).all()
    assert np.isclose(time_vector_up + time_vector_down, true_time_vector).all()
    true_transition_matrix = np.array(
        [[0, 0.75, 0],
         [1.0, 0, 1.0],
         [0, 0.25, 0]])
    assert np.isclose(transition_matrix, true_transition_matrix).all()
    
    # TODO: make more tests for avg_time_vector_up/down
    
    true_rate_matrix = np.array(
        [[-3.0/550.0, 3.0/550.0,         0],
         [ 3.0/550.0,-4.0/550.0, 1.0/100.0],
         [ 0,         1.0/550.0,-1.0/100.0]])
    #assert np.isclose(rate_matrix, true_rate_matrix).all()
    return

def test_compute_residence_time_linear():
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

    residence_time = milestoning.compute_residence_time(
        transition_matrix, time_vector, num_milestones, start_milestone)
    
    assert np.isclose(residence_time, 50.0)
    return

def test_make_time_profiles_linear():
    num_milestones = 11
    h = 1.0
    D = 1.0
    t = h**2/(2.0*D)
    transition_matrix = np.zeros((num_milestones, num_milestones))
    time_vector = np.zeros((num_milestones, 1))
    time_vector_up = np.zeros((num_milestones, 1))
    time_vector_down = np.zeros((num_milestones, 1))
    transition_matrix[1, 0] = 1.0
    transition_matrix[num_milestones-2, num_milestones-1] = 1.0
    for i in range(1, num_milestones-1):
        transition_matrix[i-1, i] = 0.5
        transition_matrix[i+1, i] = 0.5
    
    for i in range(num_milestones):
        time_vector[i, 0] = t
        time_vector_up[i, 0] = t
        time_vector_down[i, 0] = t
        
    transit_time_matrix = milestoning.make_time_profiles(
        transition_matrix, time_vector, time_vector_up, time_vector_down, 
        num_milestones)
    
    reference_matrix = np.zeros((num_milestones, num_milestones))
    for i in range(num_milestones):
        for j in range(num_milestones):
            dist = h*(i-j)
            reference_matrix[i,j] = dist**2 / (2.0*D)
            
    assert np.isclose(transit_time_matrix, reference_matrix).all()
    
def test_make_time_profiles_slant():
    num_milestones = 11
    h = 1.0
    D = 1.0
    A = -0.1
    J_a_dividedby_J_c = (np.exp(A*h)-1.0)/(1.0-np.exp(-A*h))
    J_c_dividedby_J_a = (1.0-np.exp(-A*h))/(np.exp(A*h)-1.0)
    P_b_to_a = 1.0/(1.0 + J_c_dividedby_J_a)
    P_b_to_c = 1.0/(1.0 + J_a_dividedby_J_c)
    t_b_to_a = utils.slant_time(0.0, -h, D, 1.0, A)
    t_b_to_c = utils.slant_time(0.0, h, D, 1.0, A)
    t = t_b_to_a * P_b_to_a + t_b_to_c * P_b_to_c
    transition_matrix = np.zeros((num_milestones, num_milestones))
    time_vector = np.zeros((num_milestones, 1))
    time_vector_up = np.zeros((num_milestones, 1))
    time_vector_down = np.zeros((num_milestones, 1))
    transition_matrix[1, 0] = 1.0
    transition_matrix[num_milestones-2, num_milestones-1] = 1.0
    for i in range(1, num_milestones-1):
        transition_matrix[i-1, i] = P_b_to_a
        transition_matrix[i+1, i] = P_b_to_c
    
    for i in range(0, num_milestones):
        time_vector[i, 0] = t
        time_vector_up[i, 0] = t_b_to_c
        time_vector_down[i, 0] = t_b_to_a
    
    transit_time_matrix = milestoning.make_time_profiles(
        transition_matrix, time_vector, time_vector_up, time_vector_down, 
        num_milestones)
    
    reference_matrix = np.zeros((num_milestones, num_milestones))
    for i in range(num_milestones):
        for j in range(num_milestones):
            dist = h*(i-j)
            reference_matrix[i,j] = utils.slant_time(0.0, dist, D, 1.0, A)
    
    assert np.isclose(transit_time_matrix, reference_matrix).all()
    return