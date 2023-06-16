"""
openmm_ramd/analyze/milestoning.py

Prepare and utilize a milestoning model for RAMD simulation to use
in a sigma-RAMD analysis.
"""

import numpy as np

def uniform_milestone_locations(one_dim_trajs, num_milestones):
    """
    Define a set of milestone locations uniformly spread.
    """
    # First, locate the minimum and maximum positions along the CV
    min_location = 9e9
    max_location = -9e9
    starting_cv_val = None
    for i, trajectory in enumerate(one_dim_trajs):
        for j, frame in enumerate(trajectory):
            [cv_val, xi_val] = frame
            if i == 0 and j == 0:
                starting_cv_val = cv_val
            if cv_val < min_location:
                min_location = cv_val
            if cv_val > max_location:
                max_location = cv_val
    
    assert min_location < max_location, \
        "The minimum must be less than the maximum."
    
    assert starting_cv_val is not None, "No starting CV value found."
    
    span = max_location - min_location
    h = span / (num_milestones + 1)
    milestones = []
    for i in range(num_milestones):
        milestone = h*(i+1)
        milestones.append(milestone)
        
    return milestones, min_location, max_location, starting_cv_val
    
def make_milestoning_model(
        xi_bin_trajs, milestones, frame_time, num_milestones):
    count_matrix = np.zeros((num_milestones, num_milestones), dtype=np.int32)
    time_vector = np.zeros((num_milestones, 1))
    
    for i, traj in enumerate(xi_bin_trajs):
        src_milestone = None
        src_time = 0.0
        new_cell_id = None
        src_time = -frame_time
        for j, frame in enumerate(traj):
            [cv_val, xi_val] = frame
            # Find what cell it's in
            old_cell_id = new_cell_id
            new_cell_id = num_milestones
            src_time += frame_time
            for k, milestone in enumerate(milestones):
                if cv_val < milestone:
                    new_cell_id = k
                    break
            if (old_cell_id is not None) and (new_cell_id != old_cell_id):
                # then a milestone transition has happened
                diff = new_cell_id - old_cell_id
                dest_milestone = new_cell_id - (diff + 1) // 2
                if src_milestone != dest_milestone:
                    if src_milestone is not None:
                        assert src_milestone != dest_milestone
                        count_matrix[dest_milestone, src_milestone] += 1
                        time_vector[src_milestone, 0] += src_time
                        
                    src_milestone = dest_milestone
                    src_time = 0.0
                
    #print("count_matrix:", count_matrix)
    #print("time_vector:", time_vector)
    
    transition_matrix = np.zeros((num_milestones, num_milestones))
    for i in range(num_milestones): # column index
        column_sum = np.sum(count_matrix[:,i])
        for j in range(num_milestones): # row index
            if column_sum == 0.0:
                transition_matrix[j,i] = 0.0
            else:
                transition_matrix[j,i] = count_matrix[j,i] / column_sum
    
    rate_matrix = np.zeros((num_milestones, num_milestones))
    for i in range(num_milestones):
        for j in range(num_milestones):
            if time_vector[j,0] == 0.0:
                rate_matrix[i,j] = 0.0
            else:
                rate_matrix[i,j] = count_matrix[i,j] / time_vector[j,0]
    
    #print("rate_matrix:", rate_matrix)
    
    return count_matrix, time_vector, transition_matrix, rate_matrix

def make_time_profiles(transition_matrix, time_vector, num_milestones):
    transit_time_matrix = np.zeros((num_milestones, num_milestones))
    identity = np.identity(num_milestones)
    for i in range(num_milestones):
        transition_matrix_with_sink = np.copy(transition_matrix[:,:])
        transition_matrix_with_sink[:,i] = 0.0
        geom_series = np.linalg.inv(identity-transition_matrix_with_sink)
        result = time_vector.T @ geom_series
        transit_time_matrix[i,:] = result[0,:]
        
    return transit_time_matrix