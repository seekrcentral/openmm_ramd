"""
units.py

Utilities for RAMD testing
"""

import numpy as np

def slant_time(a, b, D, beta, force_constant, use_abs=False):
    if use_abs:
        dist = abs(a-b)
    else:
        dist = a-b
    if np.isclose(force_constant, 0.0):
        time = dist**2/(2.0*D)
    else:
        time = ((np.exp(-beta*force_constant*dist) - 1.0)\
                /(beta*force_constant) + dist)/(beta*force_constant*D)
    return time

def slant_milestoning_model(num_milestones, A, h, D, base_count):
    J_a_dividedby_J_c = (np.exp(A*h)-1.0)/(1.0-np.exp(-A*h))
    J_c_dividedby_J_a = (1.0-np.exp(-A*h))/(np.exp(A*h)-1.0)
    P_b_to_a = 1.0/(1.0 + J_c_dividedby_J_a)
    P_b_to_c = 1.0/(1.0 + J_a_dividedby_J_c)
    t_b_to_a = slant_time(0.0, -h, D, 1.0, A)
    t_b_to_c = slant_time(0.0, h, D, 1.0, A)
    t = t_b_to_a * P_b_to_a + t_b_to_c * P_b_to_c
    transition_matrix = np.zeros((num_milestones, num_milestones))
    time_vector = np.zeros((num_milestones, 1))
    time_vector_up = np.zeros((num_milestones, 1))
    time_vector_down = np.zeros((num_milestones, 1))
    rate_matrix = np.zeros((num_milestones, num_milestones))
    count_matrix = np.zeros((num_milestones, num_milestones))
    transition_matrix[1, 0] = 1.0
    transition_matrix[num_milestones-2, num_milestones-1] = 1.0
    count_matrix[1, 0] = base_count
    count_matrix[num_milestones-2, num_milestones-1] = base_count
    for i in range(1, num_milestones-1):
        transition_matrix[i-1, i] = P_b_to_a
        transition_matrix[i+1, i] = P_b_to_c
        count_matrix[i-1, i] = base_count
        count_matrix[i+1, i] = base_count
    
    for i in range(0, num_milestones):
        time_vector[i, 0] = t
        time_vector_up[i, 0] = t_b_to_c
        time_vector_down[i, 0] = t_b_to_a
    
    time_vector[0, 0] = t_b_to_c
    time_vector[-1, 0] = t_b_to_a
    
    for i in range(num_milestones):
        for j in range(num_milestones):
            if time_vector[j,0] == 0.0:
                rate_matrix[i,j] = 0.0
            else:
                rate_matrix[i,j] = transition_matrix[i,j] / time_vector[j,0]
    
        rate_matrix[i,i] = -np.sum(rate_matrix[:,i])
    
    return transition_matrix, time_vector, time_vector_up, time_vector_down, \
        rate_matrix, count_matrix