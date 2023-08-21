"""
Given a milestoning model within an angle bin, obtain the profiles for PMF and 
diffusion, remove the slant for the angle bin from the PMF, then recompute
a milestoning model and use the PyGT package to estimate a likely residence
time for an unbiased system.
"""

import numpy as np

HIGH_ENERGY = 50.0

def make_energy_profile(stationary_probs, beta, ref_milestone_index=None):
    energy_profile = np.zeros(stationary_probs.shape)
    if ref_milestone_index is None:
        highest_stat_prob = max(stationary_probs)
    else:
        highest_stat_prob = stationary_probs[ref_milestone_index]
    
    for i, p_i_val in enumerate(stationary_probs):
        if p_i_val > 0.0:
            free_energy = -np.log(p_i_val / highest_stat_prob) / beta
        else:
            free_energy = HIGH_ENERGY
        
        if free_energy > HIGH_ENERGY:
            free_energy = HIGH_ENERGY
            
        energy_profile[i] = free_energy
    return energy_profile

def remove_slant(milestones, energy_profile, slanting_force, 
                 ref_milestone_index=0):
    unslanted_energy_profile = np.zeros(energy_profile.shape)
    for i, energy in enumerate(energy_profile):
        z = milestones[i]
        unslanted_energy_profile[i] = energy + z * slanting_force
    
    unslanted_energy_profile -= unslanted_energy_profile[ref_milestone_index]
    return unslanted_energy_profile
    
def unslant_transition_matrix_constant_approx(
        milestones, transition_matrix, avg_time_vector, slanting_force, beta):
    num_milestones = len(milestones)
    unslanted_rate_matrix = np.zeros((num_milestones, num_milestones), dtype=np.longdouble)
    unslanted_transition_matrix = np.zeros((num_milestones, num_milestones))
    unslanted_avg_time_vector = np.zeros((num_milestones, 1))
    
    # i == 0
    unslanted_transition_matrix[:,0] = transition_matrix[:,0]
    unslanted_avg_time_vector[0,0] = avg_time_vector[0,0]
    
    # i == num_milestones - 1
    unslanted_transition_matrix[:,-1] = transition_matrix[:,-1]
    unslanted_avg_time_vector[-1,0] = avg_time_vector[-1,0]
    
    # NOTE: this theory will neglect the effects of transitions beyond the
    # immediately adjacent milestones - creating errors, but is unavoidable.
    
    for i in range(1, num_milestones-1): # columns
        assert np.isclose(milestones[i+1] - milestones[i],
            milestones[i] - milestones[i-1]), "This method only works for "\
            "equally-spaced milestones - to leave avg. time unmodified."
        span = milestones[i+1] - milestones[i-1]
        
        p_a = np.sum(transition_matrix[:i,i])
        if p_a == 0.0:
            new_p_a = 0.0
            
        else:
            f_a = (1.0/p_a) - 1.0
            factor_a = np.exp(0.5 * beta * slanting_force * span)
            new_f_a = f_a * factor_a
            new_p_a = 1.0 / (1.0 + new_f_a)
        
        p_c = np.sum(transition_matrix[i+1:,i])
        if p_c == 0.0:
            new_p_c = 0.0
            
        else:
            f_c = (1.0/p_c) - 1.0
            factor_c = np.exp(0.5 * beta * slanting_force * -span)
            new_f_c = f_c * factor_c
            new_p_c = 1.0 / (1.0 + new_f_c)
        
        if new_p_a + new_p_c > 0.0:
            unslanted_transition_matrix[i-1,i] = new_p_a / (new_p_a + new_p_c)
            unslanted_transition_matrix[i+1,i] = new_p_c / (new_p_a + new_p_c)
            unslanted_avg_time_vector[i, 0] = avg_time_vector[i, 0]
    
    for i in range(num_milestones):
        for j in range(num_milestones):
            if unslanted_avg_time_vector[j,0] == 0.0:
                unslanted_rate_matrix[i,j] = 0.0
            else:
                unslanted_rate_matrix[i,j] = unslanted_transition_matrix[i,j] \
                    / unslanted_avg_time_vector[j,0]
    
    for i in range(num_milestones):
        unslanted_rate_matrix[i,i] = -np.sum(unslanted_rate_matrix[:,i]) 
    
    return unslanted_rate_matrix, unslanted_transition_matrix, \
            unslanted_avg_time_vector
