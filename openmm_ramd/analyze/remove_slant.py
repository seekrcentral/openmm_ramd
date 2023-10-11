"""
Given a milestoning model within an angle bin, obtain the profiles for PMF and 
diffusion, remove the slant for the angle bin from the PMF, then recompute
a milestoning model and use the PyGT package to estimate a likely residence
time for an unbiased system.
"""

import numpy as np
import scipy.integrate as sp_int
import matplotlib.pyplot as plt

HIGH_ENERGY = 500.0
THRESH_ENERGY = 50.0

def make_energy_profile(stationary_probs, beta, ref_milestone_index=None):
    energy_profile = np.zeros(stationary_probs.shape)
    if ref_milestone_index is None:
        highest_stat_prob = max(stationary_probs)
    else:
        highest_stat_prob = stationary_probs[ref_milestone_index]
    
    for i, p_i_val in enumerate(stationary_probs):
        if (p_i_val > 0.0) and (highest_stat_prob > 0.0):
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
            milestones[i] - milestones[i-1], atol=1e-2), "This method only works for "\
            "equally-spaced milestones - to leave avg. time unmodified."
        spans = np.zeros(num_milestones)
        new_ps = np.zeros(num_milestones)
        for j in range(num_milestones):
            if i == j:
                continue
            span = milestones[j] - milestones[i]
            p = transition_matrix[j,i]
            if p == 0.0:
                new_p = 0.0
            else:
                f = (1.0/p) - 1.0
                factor = np.exp(-beta * slanting_force * span)
                new_f = f * factor
                new_p = 1.0 / (1.0 + new_f)
            
            new_ps[j] = new_p
            
        sum_new_ps = np.sum(new_ps)
        if sum_new_ps > 0.0:
            for j in range(num_milestones):
                unslanted_transition_matrix[j,i] = new_ps[j] / sum_new_ps
                
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

def get_smoluchowski_flux(z_s, energy_span, diffusion_span, beta, forward):
    num_integration_points = len(z_s)
    h = (z_s[-1] - z_s[0]) / (num_integration_points - 1)
    if forward:
        ref_energy = energy_span[0]
    else:
        ref_energy = energy_span[-1]
    inv_stat_dist = np.zeros(num_integration_points)
    for j in range(num_integration_points):
        assert diffusion_span[j] > 0.0, "Diffusion must be positive. "\
            f"diffusion_span[{j}] = {diffusion_span[j]}"
        inv_stat_dist[j] = np.exp(beta*(energy_span[j] - ref_energy))\
            / diffusion_span[j]
    
    time_integral_span_outer = np.zeros(num_integration_points)
    for j in range(num_integration_points):
        time_integral_span_outer[j] = inv_stat_dist[j]
    
    inv_flux = sp_int.simps(time_integral_span_outer, dx=h)
    return 1.0 / inv_flux
    
def get_smoluchowski_u(z_s, energy_span, diffusion_span, beta, forward):
    num_integration_points = len(z_s)
    h = (z_s[-1] - z_s[0]) / (num_integration_points - 1)
    factor = get_smoluchowski_flux(z_s, energy_span, diffusion_span, beta, forward)
    stationary_dist = np.zeros(num_integration_points)
    inv_stat_dist = np.zeros(num_integration_points)
    for j in range(num_integration_points):
        stationary_dist[j] = np.exp(-beta*energy_span[j])
        inv_stat_dist[j] = np.exp(beta*energy_span[j])/diffusion_span[j]
    
    time_integral_span_outer = np.zeros(num_integration_points)
    for j in range(num_integration_points):
        if forward:
            time_integral_span_inner = np.zeros(num_integration_points-j)
            
            for k in range(j, num_integration_points):
                index2 = k - j
                time_integral_span_inner[index2] = inv_stat_dist[k]
            
            time_integral_span_outer[j] = stationary_dist[j] \
                * sp_int.simps(time_integral_span_inner, dx=h)
        else:
            time_integral_span_inner = np.zeros(j+1)
            for k in range(j+1):
                time_integral_span_inner[k] = inv_stat_dist[k]
            
            time_integral_span_outer[j] = stationary_dist[j] \
                * sp_int.simps(time_integral_span_inner, dx=h)
    
    int_u = factor * sp_int.simps(time_integral_span_outer, dx=h)
    u_z = factor * time_integral_span_outer
    return int_u, u_z

def get_smoluchowski_flux_two_points(
        a, b, energy_a, energy_b, diffusion_a, diffusion_b, beta, forward=True):
    """
    Find the time it will take to go from point a to point b, using the 
    1D Smoluchowski equation.
    """
    NUM_INTEGRATION_POINTS = 101
    if (energy_a > THRESH_ENERGY) or (energy_a > THRESH_ENERGY) \
            or (abs(energy_a - energy_b) > THRESH_ENERGY):
        return 0.0
    #h = (b - a) / (NUM_INTEGRATION_POINTS - 1)
    space_span = np.linspace(a, b, NUM_INTEGRATION_POINTS)
    energy_span = np.linspace(energy_a, energy_b, NUM_INTEGRATION_POINTS)
    diffusion_span = np.linspace(diffusion_a, diffusion_b, NUM_INTEGRATION_POINTS)
    
    flux = get_smoluchowski_flux(space_span, energy_span, diffusion_span, beta, 
                                 forward=forward)
    return flux

def get_smoluchowski_u_two_points(
        a, b, energy_a, energy_b, diffusion_a, diffusion_b, beta, forward=True):
    """
    Find the time it will take to go from point a to point b, using the 
    1D Smoluchowski equation.
    """
    NUM_INTEGRATION_POINTS = 101
    if (energy_a > THRESH_ENERGY) or (energy_a > THRESH_ENERGY) \
            or (abs(energy_a - energy_b) > THRESH_ENERGY):
        return 0.0, np.zeros(NUM_INTEGRATION_POINTS)
        
    #h = (b - a) / (NUM_INTEGRATION_POINTS - 1)
    space_span = np.linspace(a, b, NUM_INTEGRATION_POINTS)
    energy_span = np.linspace(energy_a, energy_b, NUM_INTEGRATION_POINTS)
    diffusion_span = np.linspace(diffusion_a, diffusion_b, NUM_INTEGRATION_POINTS)
    
    int_u, u_z = get_smoluchowski_u(space_span, energy_span, diffusion_span, beta, 
                                 forward=forward)
    return int_u, u_z


def unslant_transition_matrix_from_energy_profile(
        milestones, energy_profile, diffusion_profile, slanting_force, beta):
    num_milestones = len(milestones)
    unslanted_energy_profile = np.zeros(num_milestones)
    for i, z in enumerate(milestones):
        unslanted_energy_profile[i] = energy_profile[i] - z * slanting_force
    
    unslanted_rate_matrix = np.zeros((num_milestones, num_milestones), dtype=np.longdouble)
    unslanted_transition_matrix = np.zeros((num_milestones, num_milestones))
    unslanted_avg_time_vector = np.zeros((num_milestones, 1))
    
    #plt.plot(milestones, unslanted_energy_profile)
    #plt.ylim((0.0, 10.0))
    #plt.show()
    
    for i, z in enumerate(milestones):
        if i == 0:
            doing_a = False
            doing_c = True
        elif i == num_milestones - 1:
            doing_a = True
            doing_c = False
        else:
            doing_a = True
            doing_c = True
        
        total_flux = 0.0
        total_u = 0.0
        energy_b = unslanted_energy_profile[i]
        diffusion_b = diffusion_profile[i]
        
        b = z
        if doing_a:
            a = milestones[i-1]
            energy_a = unslanted_energy_profile[i-1]
            if abs(energy_a - energy_b) < THRESH_ENERGY:
                diffusion_a = diffusion_profile[i-1]
                flux_b_to_a = get_smoluchowski_flux_two_points(
                    a, b, energy_a, energy_b, diffusion_a, diffusion_b, beta, 
                    forward=False)
                u_b_to_a, u_z_backward = get_smoluchowski_u_two_points(
                    a, b, energy_a, energy_b, diffusion_a, diffusion_b, beta, 
                    forward=False)
                total_flux += flux_b_to_a
                total_u += u_b_to_a
            else:
                # energy difference too large - no flux
                flux_b_to_a = 0.0
            
        else:
            #time_b_to_a = 0.0
            flux_b_to_a = 0.0
            
        if doing_c:
            c = milestones[i+1]
            energy_c = unslanted_energy_profile[i+1]
            if abs(energy_c - energy_b) < THRESH_ENERGY:
                diffusion_c = diffusion_profile[i+1]
                flux_b_to_c = get_smoluchowski_flux_two_points(
                    b, c, energy_b, energy_c, diffusion_b, diffusion_c, beta, 
                    forward=True)
                u_b_to_c, u_z_forward = get_smoluchowski_u_two_points(
                    b, c, energy_b, energy_c, diffusion_b, diffusion_c, beta, 
                    forward=True)
                total_flux += flux_b_to_c
                total_u += u_b_to_c
            else:
                flux_b_to_c = 0.0
            
        else:
            #time_b_to_c = 0.0
            flux_b_to_c = 0.0
        
        # Tomorrow: make some tests for these functions to get to the bottom
        # of what is going wrong.
        
        if total_flux > 0.0:
            P_b_to_a = flux_b_to_a / total_flux
            P_b_to_c = flux_b_to_c / total_flux
            time = total_u / total_flux
            
            if doing_a:
                unslanted_rate_matrix[i-1, i] = P_b_to_a / time
                unslanted_transition_matrix[i-1, i] = P_b_to_a
            
            if doing_c:
                unslanted_rate_matrix[i+1, i] = P_b_to_c / time
                unslanted_transition_matrix[i+1, i] = P_b_to_c
            
            unslanted_rate_matrix[i, i] = -1.0 / time
            unslanted_avg_time_vector[i, 0] = time
            
        else:
            P_b_to_a = 0.0
            P_b_to_c = 0.0
            time = 0.0
        
        
    
    return unslanted_rate_matrix, unslanted_transition_matrix, \
        unslanted_avg_time_vector, unslanted_energy_profile
    