"""
openmm_ramd/analyze/milestoning.py

Prepare and utilize a milestoning model for RAMD simulation to use
in a theta-RAMD analysis.
"""

from collections import defaultdict

import numpy as np

MATRIX_EXPONENTIAL = 9999999999
DEFAULT_DIFFUSION_COEFF = 0.001
NM_PER_A = 0.1

def uniform_milestone_locations(one_dim_trajs, num_milestones, max_dist):
    """
    Define a set of milestone locations uniformly spread.
    """
    max_dist_factor = 0.9
    outer_edge = max_dist * max_dist_factor
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
    
    # There are 3 possible values for "situation":
    # 1. a forward exit (where max_location is close to max_dist
    # 2. a backward exit (where min_location is close to max_dist
    # 3. a two-path exit (where both min_location and max_location are close to
    #  max_dist
    
    if min_location > -outer_edge:
        # a forward exit
        situation = 1
        min_max_distance = outer_edge - min_location
        start_point_index = int((starting_cv_val - min_location) * (num_milestones - 1) / min_max_distance)
        
    elif max_location < outer_edge:
        # a backward exit
        situation = 2
        min_max_distance = max_location + outer_edge
        start_point_index = int((starting_cv_val + outer_edge) * (num_milestones - 1) / min_max_distance) + 1
        
    else:
        # a two-way exit
        situation = 3
        min_max_distance = 2.0 * outer_edge
        start_point_index = int((starting_cv_val + outer_edge) * (num_milestones - 1) / min_max_distance)
    
    num_points_after = num_milestones - start_point_index - 1
    num_points_before = num_milestones - num_points_after - 1
    if situation == 1:
        h = (outer_edge - starting_cv_val) / num_points_after
    elif situation == 2:
        h = (outer_edge + starting_cv_val) / num_points_before
    else: # situation == 3
        h = (outer_edge - starting_cv_val) / num_points_after
        
    first_milestone_location = starting_cv_val - num_points_before * h
    milestones = []
    
    for i in range(num_milestones):
        milestone = first_milestone_location + h*i
        milestones.append(milestone)
    
    return milestones, min_location, max_location, starting_cv_val, \
        start_point_index, h, situation

def dest_milestone_from_cell_transition(new_cell_id, old_cell_id):
    if new_cell_id == old_cell_id:
        raise Exception("new_cell_id must be different from old_cell_id")
    elif new_cell_id < old_cell_id:
        dest_milestone = new_cell_id
    elif new_cell_id > old_cell_id:
        dest_milestone = new_cell_id - 1
    return dest_milestone

def make_milestoning_model(
        xi_bin_trajs, milestones, frame_time, num_milestones,
        remove_unbalanced_frame_counts=False, debug_filename=None):
    
    frame_time = float(frame_time)
    count_matrix = np.zeros((num_milestones, num_milestones), dtype=np.int32)
    time_vector = np.zeros((num_milestones, 1))
    time_matrix = np.zeros((num_milestones, num_milestones))
    #time_vector_up = np.zeros((num_milestones, 1))
    #time_vector_down = np.zeros((num_milestones, 1))
    avg_time_vector = np.zeros((num_milestones, 1))
    avg_time_matrix = np.zeros((num_milestones, num_milestones))
    
    if debug_filename is not None:
        with open(debug_filename, "a") as f:
            f.write("src, dest, inc_time, cv_val, src_pos, dest_pos, "\
                    "traj_time\n")
    
    for i, traj in enumerate(xi_bin_trajs):
        src_milestone = None
        new_cell_id = None
        src_time = 0.0
        traj_time = 0.0
        
        # Here, I need to skip the counts and times the first time any state
        #  is visited within a trajectory.
        first_times = np.zeros(num_milestones)
        first_transition = -np.ones(num_milestones)
        first_milestone = None
        for j, frame in enumerate(traj):
            [cv_val, xi_val] = frame
            if j == 0:
                xi_val0 = xi_val
                if debug_filename is not None:
                    with open(debug_filename, "a") as f:
                        f.write(f"Force change to xi: {xi_val:.3f}\n")
                    
            assert xi_val == xi_val0, "Xi value changes during simulation. "\
                "Trajectory: {i}, Frame: {j}"
            # Find what cell it's in
            old_cell_id = new_cell_id
            new_cell_id = num_milestones
            src_time += frame_time
            traj_time += frame_time
            for k, milestone in enumerate(milestones):
                if cv_val < milestone:
                    new_cell_id = k
                    break
            if (old_cell_id is not None) and (new_cell_id != old_cell_id):
                # then a milestone transition has happened
                dest_milestone = dest_milestone_from_cell_transition(
                    new_cell_id, old_cell_id)
                if src_milestone != dest_milestone:
                    if src_milestone is not None:
                        if first_milestone is None:
                            first_milestone = src_milestone
                            
                        if first_times[dest_milestone] == 0.0:
                            first_times[dest_milestone] = src_time
                            first_transition[dest_milestone] = src_milestone
                        
                        count_matrix[dest_milestone, src_milestone] += 1
                        #if j < len(traj)-1:
                        time_matrix[dest_milestone, src_milestone] += src_time
                        if debug_filename is not None:
                            with open(debug_filename, "a") as f:
                                f.write(f"{src_milestone}, {dest_milestone}, "\
                                        f"{src_time:.3f}, {cv_val:.3f}, "\
                                        f"{milestones[src_milestone]:.3f}, "\
                                        f"{milestones[dest_milestone]:.3f}, "\
                                        f"{traj_time:.3f}\n")
                        
                    src_milestone = dest_milestone
                    src_time = 0.0
                    
            if src_milestone is not None and j < len(traj)-1:
                time_vector[src_milestone, 0] += frame_time
                #if cv_val > milestones[src_milestone]:
                #    time_vector_up[src_milestone, 0] += frame_time
                #else:
                #    time_vector_down[src_milestone, 0] += frame_time
            
        if src_milestone is not None:
            time_matrix[src_milestone-1,src_milestone] += src_time
        
        # If we are removing the unmatched counts/times, then do so
        if remove_unbalanced_frame_counts:
            last_milestone = dest_milestone
            if first_milestone is not None:
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
    
    for i in range(num_milestones): # column index
        column_sum = np.sum(count_matrix[:,i])
        if column_sum > 0.0:
            avg_time_vector[i, 0] = time_vector[i, 0] / column_sum
        for j in range(num_milestones):
            if count_matrix[j,i] > 0.0:
                avg_time_matrix[j, i] = time_matrix[j, i] / count_matrix[j,i]
    
    # TODO: replace above for-loop with functions below
    #avg_time_vector = make_avg_time_vector(num_milestones, time_vector, count_matrix)
    #avg_time_matrix = make_avg_time_matrix(num_milestones, time_matrix, count_matrix)
    
    #avg_time_vector[0, 0] *= 0.5 # times need to be cut in half for end states
    #avg_time_vector_up[0, 0] *= 0.5
    #avg_time_vector_down[0, 0] *= 0.5
    
    return count_matrix, time_vector, time_matrix, avg_time_vector, \
        avg_time_matrix

def make_transition_matrix(num_milestones, count_matrix):
    transition_matrix = np.zeros((num_milestones, num_milestones))
    for i in range(num_milestones): # column index
        column_sum = np.sum(count_matrix[:,i])
        for j in range(num_milestones): # row index
            if column_sum == 0.0:
                transition_matrix[j,i] = 0.0
            else:
                transition_matrix[j,i] = count_matrix[j,i] / column_sum
    
    return transition_matrix

def make_avg_time_vector(num_milestones, total_time_vector, count_matrix):
    avg_time_vector = np.zeros((num_milestones, 1))
    for i in range(num_milestones): # column index
        column_sum = np.sum(count_matrix[:,i])
        if column_sum > 0.0:
            avg_time_vector[i, 0] = total_time_vector[i, 0] / column_sum
    
    return avg_time_vector
    
def make_avg_time_matrix(num_milestones, total_time_matrix, count_matrix):
    avg_time_matrix = np.zeros((num_milestones, num_milestones))
    for i in range(num_milestones): # column index
        for j in range(num_milestones):
            if count_matrix[j,i] > 0.0:
                avg_time_matrix[j, i] = total_time_matrix[j, i] / count_matrix[j,i]
                
    return avg_time_matrix

def make_rate_matrix(num_milestones, transition_matrix, avg_time_vector):
    rate_matrix = np.zeros((num_milestones, num_milestones), dtype=np.longdouble)
    for i in range(num_milestones):
        for j in range(num_milestones):
            if avg_time_vector[j,0] == 0.0:
                rate_matrix[i,j] = 0.0
            else:
                rate_matrix[i,j] = transition_matrix[i,j] / avg_time_vector[j,0]
    
    for i in range(num_milestones):
        rate_matrix[i,i] = -np.sum(rate_matrix[:,i])
    
    return rate_matrix

def fill_gaps_in_matrix(matrix, normalize=False, flat=True):
    """
    Given a matrix with "gaps" - that is, a state where no flux/probability
    is flowing in any direction, it will fill in the gaps with the 
    nearest neighbor (or the average of the nearest neighbors).
    This algorithm assumes that rows are the states FROM, and columns
    are the states TO.
    """
    zero_column_indices = []
    for m in range(matrix.shape[1]):
        # loop thru columns
        if m > 0:
            column_sum_down = np.sum(matrix[:m,m])
        else:
            column_sum_down = 0.0
            
        if m < matrix.shape[1] - 1:
            column_sum_up = np.sum(matrix[m+1:,m])
        else:
            column_sum_up = 0.0
            
        column_sum = column_sum_down + column_sum_up
        if column_sum == 0.0: 
            # then we need to find a neighbor to 'steal' stats from
            zero_column_indices.append(m)
        
    replacement_dict = defaultdict(list)
    for zero_column_index in zero_column_indices:
        found_replacement = False
        for interval in range(1, matrix.shape[1]):
            upper_index = zero_column_index - interval
            lower_index = zero_column_index + interval
            if (upper_index >= 0) and (upper_index not in zero_column_indices):
                found_replacement = True
                replacement_dict[zero_column_index].append(upper_index)
                
            if (lower_index < matrix.shape[1]) \
                    and (lower_index not in zero_column_indices):
                found_replacement = True
                replacement_dict[zero_column_index].append(lower_index)
            
            if found_replacement:
                break
        
    for col in replacement_dict:
        other_list = replacement_dict[col]
        num_others = len(other_list)
        if num_others == 0:
            continue
        for other in other_list:
            if matrix[other, other] == 0.0:
                value = 1.0
            else:
                value = -matrix[other, other]
                
            if (col > 0) and (other > 0):
                if flat:
                    matrix[col-1, col] += value / num_others
                else:
                    matrix[col-1, col] += matrix[other-1, other] / num_others
                
            if (col < matrix.shape[1]-1) and (other < matrix.shape[1]-1):
                if flat:
                    matrix[col+1, col] += value / num_others
                else:
                    matrix[col+1, col] += matrix[other+1, other] / num_others
                
    if normalize:
        for m in range(matrix.shape[1]):
            column_sum = np.sum(matrix[:, m])
            if column_sum > 0.0:
                matrix[:, m] /= column_sum
    
    return

def fill_gaps_in_vector(vector):
    """
    Given a vector with "gaps" - that is, a state where no value
    is in the state, it will fill in the gaps with the 
    nearest neighbor (or the average of the nearest neighbors).
    """
    n = len(vector)
    zero_indices = []
    for i, entry in enumerate(vector):
        if entry == 0.0: 
            # then we need to find a neighbor to 'steal' stats from
            zero_indices.append(i)
        
    replacement_dict = defaultdict(list)
    for zero_index in zero_indices:
        found_replacement = False
        for interval in range(1, n):
            upper_index = zero_index - interval
            lower_index = zero_index + interval
            if (upper_index >= 0) and (upper_index not in zero_indices):
                found_replacement = True
                replacement_dict[zero_index].append(upper_index)
                
            if (lower_index < n) \
                    and (lower_index not in zero_indices):
                found_replacement = True
                replacement_dict[zero_index].append(lower_index)
            
            if found_replacement:
                break
        
    for i in replacement_dict:
        other_list = replacement_dict[i]
        num_others = len(other_list)
        if num_others == 0:
            continue
        for other in other_list:
            vector[i] += vector[other] / num_others
    
    return

def compute_stationary_probabilities(transition_matrix, avg_time_vector, 
                                     num_milestones, start_index, situation,
                                     wrap_sink=True):
    TINY_PROB = 1e-6
    #initial_prob_vector = np.zeros((1, num_milestones))
    #initial_prob_vector[0,0] = 1.0
    initial_prob_vector = np.ones((1, num_milestones))
    transition_matrix_with_leak = np.copy(transition_matrix)
    if (situation == 1) or (situation == 3):
        if transition_matrix_with_leak[-2, -1] < TINY_PROB:
            transition_matrix_with_leak[-2, -1] = 1.0
            avg_time_vector[-1, 0] = avg_time_vector[-2, 0]
        
        transition_matrix_with_leak[-1, -1] += TINY_PROB
        transition_matrix_with_leak[-2, -1] -= TINY_PROB
        if wrap_sink:
            transition_matrix_with_leak[:, -1] = 0.0
            transition_matrix_with_leak[start_index, -1] = 1.0
            
    if (situation == 2) or (situation == 3):
        if transition_matrix_with_leak[1, 0] < TINY_PROB:
            transition_matrix_with_leak[1, 0] = 1.0
            avg_time_vector[0, 0] = avg_time_vector[1, 0]
        
        transition_matrix_with_leak[0, 0] += TINY_PROB
        transition_matrix_with_leak[1, 0] -= TINY_PROB
        if wrap_sink:
            transition_matrix_with_leak[:, 0] = 0.0
            transition_matrix_with_leak[start_index, 0] = 1.0
    
    if situation == 1:
        if transition_matrix_with_leak[1, 0] > TINY_PROB:
            transition_matrix_with_leak[0, 0] += TINY_PROB
            transition_matrix_with_leak[1, 0] -= TINY_PROB
    
    elif situation == 2:
        if transition_matrix_with_leak[-2, -1] > TINY_PROB:
            transition_matrix_with_leak[-1, -1] += TINY_PROB
            transition_matrix_with_leak[-2, -1] -= TINY_PROB
    
    for i in range(1, num_milestones-1):
        if (transition_matrix_with_leak[i-1, i] > TINY_PROB) \
                and (transition_matrix_with_leak[i+1, i] > TINY_PROB):
            transition_matrix_with_leak[i, i] += TINY_PROB
            transition_matrix_with_leak[i-1, i] -= 0.5*TINY_PROB
            transition_matrix_with_leak[i+1, i] -= 0.5*TINY_PROB
        else:
            transition_matrix_with_leak[i-1, i] += TINY_PROB
            transition_matrix_with_leak[i+1, i] += TINY_PROB
            column_sum = np.sum(transition_matrix_with_leak[:, i])
            transition_matrix_with_leak[:, i] /= column_sum
    
    q_i = initial_prob_vector @ np.linalg.matrix_power(
        transition_matrix_with_leak.T, MATRIX_EXPONENTIAL)    
    if (np.sum(q_i) > 0.0) and np.isfinite(np.sum(q_i)):
        q_i = q_i / np.sum(q_i)
    
    p_i = np.zeros(num_milestones)
    for i in range(num_milestones):
        if i == 0 or i == num_milestones - 1:
        #if i == num_milestones - 1:
            # This is because at the ends, the reflective boundary conditions
            #  cause the flux to be undercounted by a factor of 2
            flux_factor = 2.0
        else:
            flux_factor = 1.0
            
        p_i[i] = q_i[0, i] * avg_time_vector[i, 0] * flux_factor
    
    sum_p_i = np.sum(p_i)
    if sum_p_i > 0.0:
        for i in range(num_milestones):
            p_i[i] /= sum_p_i
    
    return p_i

def compute_diffusion_coefficients(
        rate_matrix, transition_matrix, free_energy_profile, milestones, beta):
    num_milestones = len(milestones)
    diffusion_coeffs = np.zeros(num_milestones)
    for i in range(num_milestones):
        if i == 0:
            if rate_matrix[1, 0] == 0.0:
                diffusion_coeffs[0] = DEFAULT_DIFFUSION_COEFF
                continue
                
            incubation_time = 1.0 / rate_matrix[1, 0]
            span = NM_PER_A * (milestones[1] - milestones[0])
            energy_diff = (free_energy_profile[1] - free_energy_profile[0])
            diffusion_coeffs[0] = 0.5 * span**2 * np.exp(beta*energy_diff/3.0) \
                / incubation_time

        elif i == num_milestones - 1:
            last = num_milestones - 1
            if rate_matrix[last-1, last] == 0.0:
                diffusion_coeffs[last] = DEFAULT_DIFFUSION_COEFF
                continue
            
            incubation_time = 1.0 / rate_matrix[last-1, last]
            span = NM_PER_A * (milestones[last] - milestones[last-1])
            energy_diff = (free_energy_profile[last] \
                           - free_energy_profile[last-1])
            if rate_matrix[last-1, last] == 0.0:
                diffusion_coeffs[last] = diffusion_coeffs[last-1]
            else:
                diffusion_coeffs[last] = 0.5 * span**2 \
                    * np.exp(-beta*energy_diff/3.0) / incubation_time
            
        else:
            prob1 = sum(transition_matrix[:i, i])
            prob2 = sum(transition_matrix[i+1:, i])
            if not np.isclose(prob1 + prob2, 1.0):
                diffusion_coeffs[i] == DEFAULT_DIFFUSION_COEFF
                continue
                
            if rate_matrix[i,i] == 0.0:
                diffusion_coeffs[i] = DEFAULT_DIFFUSION_COEFF
                continue
            
            incubation_time = -1.0 / rate_matrix[i,i]
            span1 = NM_PER_A * (milestones[i] - milestones[i-1])
            span2 = NM_PER_A * (milestones[i+1] - milestones[i])
            energy_diff1 = (free_energy_profile[i] - free_energy_profile[i-1])
            energy_diff2 = (free_energy_profile[i+1] - free_energy_profile[i])
            diffusion_coeffs[i] += 0.5 * prob1 * span1**2 \
                * np.exp(-beta*energy_diff1/3.0) / incubation_time
            diffusion_coeffs[i] += 0.5 * prob2 * span2**2 \
                * np.exp(beta*energy_diff2/3.0) / incubation_time
        
        if diffusion_coeffs[i] == 0.0:
            diffusion_coeffs[i] = DEFAULT_DIFFUSION_COEFF
    return diffusion_coeffs

# The old way of doing diffusion coeffs
def compute_diffusion_coefficients_OLD(rate_matrix, num_milestones, milestone_spacing_in_A):
    diffusion_coeffs = np.zeros(num_milestones)
    milestone_spacing = 0.1 * milestone_spacing_in_A
    for i in range(num_milestones):
        if i == 0:
            diffusion_coeffs[0] = 0.5 * milestone_spacing**2 \
                * rate_matrix[1, 0]

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
        
        if diffusion_coeffs[i] == 0.0:
            diffusion_coeffs[i] = DEFAULT_DIFFUSION_COEFF
    return diffusion_coeffs

def compute_counts(count_matrix, num_milestones):
    LOWER_LIMIT = 2 # could also be 1
    counts = np.zeros(num_milestones)
    for i in range(num_milestones):
        column_sum = np.sum(count_matrix[:,i])
        # This was the old way of doing it
        #counts[i] = column_sum
        # But this was problematic, because if a transition were ever one-way
        # Then there would be a huge jump in potential, which would be bad. So
        # If either side of the matrix sums to 0, then we should set the 
        # count matrix to zero also.
        if (np.sum(count_matrix[:i,i]) < LOWER_LIMIT) \
                or (np.sum(count_matrix[(i+1):,i]) < LOWER_LIMIT):
            counts[i] = 0
        else:
            counts[i] = column_sum
            
    return counts