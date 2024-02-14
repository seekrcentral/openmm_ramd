"""
test_samples.py

"""
import numpy as np

import openmm_ramd.analyze.modules.samples as samples

def test_combine_dest_states():
    Q = np.array([[-1.1, 1.1, 0.0, 0.0],
                  [1.2, -2.5, 1.3, 0.0],
                  [0.0, 1.4, -2.9, 1.5],
                  [0.0, 0.0, 1.6, -1.6]])
    dest_indices = [3]
    old_src_indices = [1,2]
    newQ, new_src_indices, dest_index = samples.combine_dest_states(
        Q, old_src_indices, dest_indices)
    assert dest_index == 3
    assert np.isclose(Q, newQ).all()
    assert new_src_indices == [1,2]
    
    dest_indices2 = [0, 3]
    newQ2, new_src_indices2, dest_index2 = samples.combine_dest_states(
        Q, old_src_indices, dest_indices2)
    assert dest_index2 == 2
    trueQ = np.array([[-2.5, 1.3, 1.2],
                      [1.4, -2.9, 1.5],
                      [1.1, 1.6, -2.7]])
    assert np.isclose(trueQ, newQ2).all()
    assert new_src_indices2 == [0,1]
    return

def test_compute_time_flat_one_and_two_sinks():
    num_milestones = 11
    a = 0.0
    b = 10.0
    milestones = np.linspace(a, b, num_milestones)
    diffusion_coeff = 1.0
    
    flat_rate_matrix = np.zeros((num_milestones, num_milestones))
    
    span = milestones[1] - milestones[0]
    time_forward_flat = span / (2.0 * diffusion_coeff)
    time_backward_flat = time_forward_flat
    
    for i in range(num_milestones):
        if i == 0:
            flat_rate_matrix[i+1, i] = 1.0/time_forward_flat
            
        elif i == num_milestones-1:
            flat_rate_matrix[i-1, i] = 1.0/time_backward_flat
            
        else:
            flat_rate_matrix[i-1, i] = 0.5/time_forward_flat
            flat_rate_matrix[i+1, i] = 0.5/time_forward_flat
            
    time = samples.compute_time(flat_rate_matrix, 0, [10])
    assert np.isclose(time, (b-a)**2/(2.0*diffusion_coeff))
    
    num_milestones2 = 21
    a = -10.0
    b = 10.0
    milestones2 = np.linspace(a, b, num_milestones2)
    flat_rate_matrix2 = np.zeros((num_milestones2, num_milestones2))
    
    span2 = milestones2[1] - milestones2[0]
    time_forward_flat2 = span2 / (2.0 * diffusion_coeff)
    time_backward_flat2 = time_forward_flat2
    for i in range(num_milestones2):
        if i == 0:
            flat_rate_matrix2[i+1, i] = 1.0/time_forward_flat2
            
        elif i == num_milestones2-1:
            flat_rate_matrix2[i-1, i] = 1.0/time_backward_flat2
            
        else:
            flat_rate_matrix2[i-1, i] = 0.5/time_forward_flat2
            flat_rate_matrix2[i+1, i] = 0.5/time_forward_flat2
            
    time2 = samples.compute_time(flat_rate_matrix2, 10, [0, 20])
    assert np.isclose(time2, (b-0)**2/(2.0*diffusion_coeff))
    