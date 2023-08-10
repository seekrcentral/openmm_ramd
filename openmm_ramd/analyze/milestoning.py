"""
openmm_ramd/analyze/milestoning.py

Prepare and utilize a milestoning model for RAMD simulation to use
in a sigma-RAMD analysis.
"""

import numpy as np
from scipy import linalg as la

MATRIX_EXPONENTIAL = 9999999999

def uniform_milestone_locations(one_dim_trajs, num_milestones, maxDist):
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
    
    min_max_distance = maxDist - min_location
    start_point_index = int((starting_cv_val - min_location) * (num_milestones - 1) / min_max_distance)
    num_points_after = num_milestones - start_point_index - 1
    num_points_before = num_milestones - num_points_after - 1
    h = (maxDist - starting_cv_val) / num_points_after
    first_milestone_location = starting_cv_val - num_points_before * h
    milestones = [] #[start_milestone_location]
    #for i in range(num_milestones-1):
    
    # DEBUG
    #first_milestone_location = 0.0
    #h = 1.0
    
    for i in range(num_milestones):
        milestone = first_milestone_location + h*i
        milestones.append(milestone)
    
    #starting_index = 0
    return milestones, min_location, max_location, starting_cv_val, \
        start_point_index, h
    
def make_milestoning_model(
        xi_bin_trajs, milestones, frame_time, num_milestones,
        remove_unbalanced_frame_counts=True):
    
    frame_time = float(frame_time)
    count_matrix = np.zeros((num_milestones, num_milestones), dtype=np.int32)
    time_vector = np.zeros((num_milestones, 1))
    time_vector_up = np.zeros((num_milestones, 1))
    time_vector_down = np.zeros((num_milestones, 1))
    avg_time_vector = np.zeros((num_milestones, 1))
    avg_time_vector_up = np.zeros((num_milestones, 1))
    avg_time_vector_down = np.zeros((num_milestones, 1))
    
    for i, traj in enumerate(xi_bin_trajs):
        src_milestone = None
        new_cell_id = None
        src_time = 0.0
        #src_time_above = 
        #src_time_below = 
        
        # Here, I need to skip the counts and times the first time any state
        #  is visited within a trajectory.
        first_times = np.zeros(num_milestones)
        first_transition = -np.ones(num_milestones)
        first_milestone = None
        
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
                        if first_milestone is None:
                            first_milestone = src_milestone
                            
                        if first_times[dest_milestone] == 0.0:
                            first_times[dest_milestone] = src_time
                            first_transition[dest_milestone] = src_milestone
                            
                        count_matrix[dest_milestone, src_milestone] += 1
                        #time_vector[src_milestone, 0] += recorded_time
                        #if dest_milestone > src_milestone:
                        #    time_vector_up[src_milestone, 0] += recorded_time
                        #else:
                        #    time_vector_down[src_milestone, 0] += recorded_time
                        
                    src_milestone = dest_milestone
                    src_time = 0.0
                    
            if src_milestone is not None and j < len(traj)-1:
                time_vector[src_milestone, 0] += frame_time
                if cv_val > milestones[src_milestone]:
                    time_vector_up[src_milestone, 0] += frame_time
                else:
                    time_vector_down[src_milestone, 0] += frame_time
        
        # If we are removing the unmatched counts/times, then do so
        if remove_unbalanced_frame_counts:
            last_milestone = dest_milestone
            
            if i == -1:
                print("first_times")
                print(first_times)
                print("first_transition")
                print(first_transition)
                print("first_milestone:", first_milestone)
                print("last_milestone:", last_milestone)
                
                print("count_matrix before:")
                print(count_matrix)
                print("time_vector before:")
                print(time_vector)
            
            if first_milestone+1 < last_milestone:
                for j in range(first_milestone+1, last_milestone+1):
                    if first_transition[j] < 0:
                        continue
                    src = int(first_transition[j])
                    count_matrix[j, src] -= 1
                    first_time = first_times[j]
                    time_vector[src, 0] -= first_time
            
            if last_milestone+1 < first_milestone:
                for j in range(last_milestone, first_milestone):
                    if first_transition[j] < 0:
                        continue
                    src = int(first_transition[j])
                    count_matrix[j, src] -= 1
                    first_time = first_times[j]
                    time_vector[src, 0] -= first_time
                
                    # Uncomment someday to check backwards transitions
            
            if i == -1:
                print("count_matrix after:")
                print(count_matrix)
                print("time_vector before:")
                print(time_vector)
                exit()
            
    
    transition_matrix = np.zeros((num_milestones, num_milestones))
    for i in range(num_milestones): # column index
        column_sum = np.sum(count_matrix[:,i])
        if column_sum > 0.0:
            avg_time_vector[i, 0] = time_vector[i, 0] / column_sum
        column_sum_up = np.sum(count_matrix[i+1:,i])
        if column_sum_up > 0.0:
            avg_time_vector_up[i, 0] = time_vector_up[i, 0] / column_sum_up
        column_sum_down = np.sum(count_matrix[:i,i])
        if column_sum_down > 0.0:
            avg_time_vector_down[i, 0] = time_vector_down[i, 0] / column_sum_down
        
        for j in range(num_milestones): # row index
            if column_sum == 0.0:
                transition_matrix[j,i] = 0.0
            else:
                transition_matrix[j,i] = count_matrix[j,i] / column_sum
                
    avg_time_vector[0, 0] *= 0.5 # times need to be cut in half for end states
    avg_time_vector_up[0, 0] *= 0.5
    avg_time_vector_down[0, 0] *= 0.5
    
    rate_matrix = np.zeros((num_milestones, num_milestones))
    for i in range(num_milestones):
        for j in range(num_milestones):
            if avg_time_vector[j,0] == 0.0:
                rate_matrix[i,j] = 0.0
            else:
                rate_matrix[i,j] = transition_matrix[i,j] / avg_time_vector[j,0]
    
    for i in range(num_milestones):
        rate_matrix[i,i] = -np.sum(rate_matrix[:,i])
        
    return count_matrix, time_vector, time_vector_up, time_vector_down, \
        transition_matrix, avg_time_vector, avg_time_vector_up, \
        avg_time_vector_down, rate_matrix

def compute_stationary_probabilities(transition_matrix, avg_time_vector, 
                                     num_milestones, start_index, 
                                     wrap_sink=True):
    TINY_PROB = 1e-6
    initial_prob_vector = np.zeros((1, num_milestones))
    initial_prob_vector[0,0] = 1.0
    
    transition_matrix_with_leak = np.copy(transition_matrix)
    if transition_matrix_with_leak[-2, -1] < TINY_PROB:
        transition_matrix_with_leak[-2, -1] = 1.0
        avg_time_vector[-1, 0] = avg_time_vector[-2, 0]
    
    transition_matrix_with_leak[-1, -1] += TINY_PROB
    transition_matrix_with_leak[-2, -1] -= TINY_PROB
    
    if wrap_sink:
        transition_matrix_with_leak[:, -1] = 0.0
        transition_matrix_with_leak[start_index, -1] = 1.0
    
    transition_matrix_with_leak[0, 0] += TINY_PROB
    transition_matrix_with_leak[1, 0] -= TINY_PROB
    
    for i in range(1, num_milestones-1):
        transition_matrix_with_leak[i, i] += TINY_PROB
        transition_matrix_with_leak[i-1, i] -= 0.5*TINY_PROB
        transition_matrix_with_leak[i+1, i] -= 0.5*TINY_PROB
    
    q_i = initial_prob_vector @ np.linalg.matrix_power(
        transition_matrix_with_leak.T, MATRIX_EXPONENTIAL)
    q_i = q_i / np.sum(q_i)
    p_i = np.zeros(num_milestones)
    for i in range(num_milestones):
        if i == 0 or i == num_milestones - 1:
            # This is because at the ends, the reflective boundary conditions
            #  cause the flux to be undercounted by a factor of 2
            flux_factor = 1.0
        else:
            flux_factor = 1.0
            
        p_i[i] = q_i[0, i] * avg_time_vector[i, 0] * flux_factor
    
    sum_p_i = np.sum(p_i)
    for i in range(num_milestones):
        p_i[i] /= sum_p_i
        
    return p_i

def compute_diffusion_coefficients(rate_matrix, num_milestones, milestone_spacing):
    diffusion_coeffs = np.zeros(num_milestones)
    for i in range(num_milestones):
        if i == 0:
            diffusion_coeffs[0] = 0.5 * milestone_spacing**2 \
                * rate_matrix[1, 0]
            #diffusion_coeffs[0] = milestone_spacing**2 \
            #    * rate_matrix[1, 0]

        elif i == num_milestones - 1:
            last = num_milestones - 1
            if rate_matrix[last-1, last] == 0.0:
                diffusion_coeffs[last] = diffusion_coeffs[last-1]
            else:
                diffusion_coeffs[last] = 0.5 * milestone_spacing**2 \
                    * rate_matrix[last-1, last]
            
        else:
            diffusion_coeffs[i] += 0.5 * milestone_spacing**2 \
                * rate_matrix[i+1, i]
            diffusion_coeffs[i] += 0.5 * milestone_spacing**2 \
                * rate_matrix[i-1, i]
                
    return diffusion_coeffs

def compute_counts(count_matrix, num_milestones):
    counts = np.zeros(num_milestones)
    for i in range(num_milestones):
        counts[i] = np.sum(count_matrix[:,i])
    
    return counts

def compute_residence_time(transition_matrix, avg_time_vector, num_milestones,
                           start_milestone):
    if np.sum(transition_matrix[-1,:]) == 0.0:
        return np.inf
    identity = np.identity(num_milestones)
    transition_matrix_with_sink = np.copy(transition_matrix)
    transition_matrix_with_sink[-1,:] = 0.0
    transition_matrix_with_sink[:,-1] = 0.0
    I_minus_K = identity-transition_matrix_with_sink
    try:
        geom_series = np.linalg.inv(I_minus_K)
    except np.linalg.LinAlgError:
        return np.inf
    
    rate_matrix = geom_series # - identity
    times = avg_time_vector.T @ rate_matrix
    residence_time = times[0, start_milestone]
    return residence_time

def make_time_profiles(transition_matrix, avg_time_vector, avg_time_vector_up, 
                       avg_time_vector_down, num_milestones):
    transit_time_matrix = np.zeros((num_milestones, num_milestones))
    identity = np.identity(num_milestones)
    
    time_vector = np.zeros((num_milestones, 1))
    time_vector[0, 0] = avg_time_vector_up[0, 0]
    #time_vector[0, 0] = avg_time_vector[0, 0]
    time_vector[num_milestones-1, 0] = avg_time_vector_down[num_milestones-1, 0]
    for i in range(1, num_milestones-1):
        #P_up = transition_matrix[i+1, i]
        #P_down = transition_matrix[i-1, i]
        #time_vector[i, 0] = P_up * avg_time_vector_up[i, 0] + P_down \
        #    * avg_time_vector_down[i, 0]
        time_vector[i, 0] = avg_time_vector[i, 0]
    
    for i in range(num_milestones):
        for j in range(num_milestones):
            if i == j:
                transit_time_matrix[i, j] = 0.0
            else:
                if np.sum(transition_matrix[i,:]) == 0.0:
                    transit_time_matrix[i,:] = np.inf
                    continue
                
                transition_matrix_with_sink = np.copy(transition_matrix[:,:])
                time_vector_adjusted = np.copy(time_vector[:,:])
                transition_matrix_with_sink[i,:] = 0.0
                transition_matrix_with_sink[:,i] = 0.0
                # Prevent probability from flowing away from the sink
                if (i < j):
                    # It cannot flow 'forwards': the sink is behind
                    # Deactivate all states up to the sink
                    transition_matrix_with_sink[:,:i] = 0.0
                    if j < num_milestones - 1: 
                        transition_matrix_with_sink[:,j] = 0.0
                        if j-1 != i:
                            transition_matrix_with_sink[j-1,j] = 1.0
                        time_vector_adjusted[j, 0] = avg_time_vector_down[j, 0]
                        time_vector_adjusted[j+1:, 0] = 0.0
                    
                elif (i > j): 
                    # It cannot flow "backwards": the sink is ahead
                    # Deactivate all states beyond the sink
                    transition_matrix_with_sink[:,i:] = 0.0
                    if j > 0:
                        transition_matrix_with_sink[:,j] = 0.0
                        if j+1 != i:
                            transition_matrix_with_sink[j+1,j] = 1.0
                        time_vector_adjusted[j, 0] = avg_time_vector_up[j, 0]
                        time_vector_adjusted[:j, 0] = 0.0
                
                I_minus_K = identity-transition_matrix_with_sink
                try:
                    geom_series = np.linalg.inv(I_minus_K)
                    # Remove self-time
                    rate_matrix = geom_series # - identity
                    result = time_vector_adjusted.T @ rate_matrix
                    if result[0,j] < 0.0: 
                        print("transition_matrix_with_sink:")
                        print(transition_matrix_with_sink)
                        print("time_vector_adjusted:")
                        print(time_vector_adjusted)
                        raise Exception("Negative time detected.")
                    transit_time_matrix[i,j] = result[0,j]
                except np.linalg.LinAlgError:
                    transit_time_matrix[i,j] = np.inf
    
    return transit_time_matrix