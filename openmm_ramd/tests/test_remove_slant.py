"""
test_remove_slant.py
"""

import numpy as np
import scipy.integrate as sp_int

import openmm_ramd.analyze.modules.remove_slant as remove_slant

def test_get_smoluchowski_flux():
    a = 1.1
    b = 1.2
    c = 1.3
    energy_a = 1.0
    energy_b = 2.0
    energy_c = 4.0
    diffusion = 1.0
    beta = 0.5
    
    A_1 = (energy_b - energy_a) / (b - a)
    A_2 = (energy_c - energy_b) / (c - b)
    
    NUM_INTEGRATION_POINTS = 101
    z_s1 = np.linspace(a, b, NUM_INTEGRATION_POINTS)
    energy_span1 = np.linspace(energy_a, energy_b, NUM_INTEGRATION_POINTS)
    diffusion_span = np.linspace(diffusion, diffusion, NUM_INTEGRATION_POINTS)
    flux1 = remove_slant.get_smoluchowski_flux(
        z_s1, energy_span1, diffusion_span, beta, forward=False)
    
    exact_flux1 = beta * A_1 * diffusion / (1.0 - np.exp(-beta*A_1*(b-a)))
    assert np.isclose(flux1, exact_flux1)
    
    flux_two_points1 = remove_slant.get_smoluchowski_flux_two_points(
        a, b, energy_a, energy_b, diffusion, diffusion, beta, forward=False)
    assert np.isclose(flux_two_points1, exact_flux1)
    
    z_s2 = np.linspace(b, c, NUM_INTEGRATION_POINTS)
    energy_span2 = np.linspace(energy_b, energy_c, NUM_INTEGRATION_POINTS)
    flux2 = remove_slant.get_smoluchowski_flux(
        z_s2, energy_span2, diffusion_span, beta, forward=True)
    
    exact_flux2 = beta * A_2 * diffusion / (np.exp(beta*A_2*(c-b)) - 1.0)
    assert np.isclose(flux2, exact_flux2)
    
    flux_two_points2 = remove_slant.get_smoluchowski_flux_two_points(
        b, c, energy_b, energy_c, diffusion, diffusion, beta, forward=True)
    assert np.isclose(flux_two_points2, exact_flux2)
    return

def test_get_smoluchowski_u():
    a = 1.1
    b = 1.2
    c = 1.3
    energy_a = 1.0
    energy_b = 2.0
    energy_c = 4.0
    diffusion = 1.0
    beta = 0.5
    
    A_1 = (energy_b - energy_a) / (b - a)
    A_2 = (energy_c - energy_b) / (c - b)
    
    NUM_INTEGRATION_POINTS = 101
    z_s1 = np.linspace(a, b, NUM_INTEGRATION_POINTS)
    energy_span1 = np.linspace(energy_a, energy_b, NUM_INTEGRATION_POINTS)
    diffusion_span = np.linspace(diffusion, diffusion, NUM_INTEGRATION_POINTS)
    
    int_u1, u_z1 = remove_slant.get_smoluchowski_u(
        z_s1, energy_span1, diffusion_span, beta, forward=False)
    
    factor1 = 1.0 / (1.0 - np.exp(-beta*A_1*(b-a)))
    exact_int_u1 = factor1 * (b - a - (1.0 - np.exp(-beta*A_1*(b-a)))/(beta*A_1))
    assert np.isclose(int_u1, exact_int_u1)
    
    int_u_two_points1, u_z_two_points1 = remove_slant.get_smoluchowski_u_two_points(
        a, b, energy_a, energy_b, diffusion, diffusion, beta, forward=False)
    
    assert np.isclose(int_u_two_points1, exact_int_u1)
    
    z_s2 = np.linspace(b, c, NUM_INTEGRATION_POINTS)
    energy_span2 = np.linspace(energy_b, energy_c, NUM_INTEGRATION_POINTS)
    
    int_u2, u_z2 = remove_slant.get_smoluchowski_u(
        z_s2, energy_span2, diffusion_span, beta, forward=True)
    
    factor2 = 1.0 / (np.exp(beta*A_2*(c-b)) - 1.0)
    exact_int_u2 = factor2 * ((np.exp(beta*A_2*(c-b)) - 1.0)/(beta*A_2) - c + b)
    assert np.isclose(int_u2, exact_int_u2)
    
    int_u_two_points2, u_z_two_points2 = remove_slant.get_smoluchowski_u_two_points(
        b, c, energy_b, energy_c, diffusion, diffusion, beta, forward=True)
    
    assert np.isclose(int_u_two_points2, exact_int_u2)
    return

def test_unslant_transition_matrix_constant_approx():
    num_milestones = 11
    milestones = np.linspace(0.0, 10.0, num_milestones)
    slanting_force = -0.1
    diffusion_coeff = 1.0
    beta = 0.5
    
    flat_rate_matrix = np.zeros((num_milestones, num_milestones))
    flat_transition_matrix = np.zeros((num_milestones, num_milestones))
    flat_avg_time_vector = np.zeros((num_milestones, 1))
    flat_avg_time_matrix = np.zeros((num_milestones, num_milestones))
    
    span = milestones[1] - milestones[0]
    time_forward_flat = span / (2.0 * diffusion_coeff)
    time_backward_flat = time_forward_flat
    
    for i in range(num_milestones):
        if i == 0:
            flat_transition_matrix[i+1, i] = 1.0
            flat_avg_time_vector[i, 0] = time_forward_flat
            flat_avg_time_matrix[i+1, i] = time_forward_flat
            flat_rate_matrix[i+1, i] = 1.0/flat_avg_time_vector[i, 0]
            
        elif i == num_milestones-1:
            flat_transition_matrix[i-1, i] = 1.0
            flat_avg_time_vector[i, 0] = time_backward_flat
            flat_avg_time_matrix[i-1, i] = time_backward_flat
            flat_rate_matrix[i-1, i] = 1.0/flat_avg_time_vector[i, 0]
            
        else:
            flat_transition_matrix[i-1, i] = 0.5
            flat_transition_matrix[i+1, i] = 0.5
            flat_avg_time_vector[i, 0] = time_forward_flat
            flat_avg_time_matrix[i-1, i] = time_backward_flat
            flat_avg_time_matrix[i+1, i] = time_forward_flat
            flat_rate_matrix[i-1, i] = 0.5/flat_avg_time_vector[i, 0]
            flat_rate_matrix[i+1, i] = 0.5/flat_avg_time_vector[i, 0]
    
    #print("flat_transition_matrix:")
    #print(flat_transition_matrix)
    #print("flat_avg_time_vector:")
    #print(flat_avg_time_vector)
    #print("flat_avg_time_matrix:")
    #print(flat_avg_time_matrix)
    #print("flat_rate_matrix:")
    #print(flat_rate_matrix)
    
    unslanted_rate_matrix, unslanted_transition_matrix, \
        unslanted_avg_time_vector \
        = remove_slant.unslant_transition_matrix_constant_approx(
            milestones, flat_transition_matrix, flat_avg_time_vector, 
            flat_avg_time_matrix, slanting_force, beta)
    
    #print("unslanted_rate_matrix:")
    #print(unslanted_rate_matrix)
    #print("unslanted_transition_matrix:")
    #print(unslanted_transition_matrix)
    #print("unslanted_avg_time_vector:")
    #print(unslanted_avg_time_vector)
    
    A = slanting_force
    diffusion = diffusion_coeff
    a = milestones[0]
    b = milestones[1]
    c = milestones[2]
    
    factor1 = 1.0 / (1.0 - np.exp(-beta*A*(c-b)))
    second_milestone_flux1 = beta * A * diffusion * factor1
    second_milestone_int_u1 = factor1 * (c - b - (1.0 - np.exp(-beta*A*(c-b)))/(beta*A))
    
    factor2 = 1.0 / (np.exp(beta*A*(c-b)) - 1.0)
    second_milestone_flux2 = beta * A * diffusion * factor2
    second_milestone_int_u2 = factor2 * ((np.exp(beta*A*(c-b)) - 1.0)/(beta*A) - c + b)
    
    second_milestone_time = (second_milestone_int_u1 + second_milestone_int_u2) \
        / (second_milestone_flux1 + second_milestone_flux2)
    
    p_down = second_milestone_flux2 / (second_milestone_flux1 + second_milestone_flux2)
    p_up = second_milestone_flux1 / (second_milestone_flux1 + second_milestone_flux2)
    rate_down = p_down/second_milestone_time
    rate_up = p_up/second_milestone_time
    
    #print("second_milestone_time:", second_milestone_time)
    #print("p_down:", p_down)
    #print("p_up:", p_up)
    #print("rate_down:", rate_down)
    #print("rate_up:", rate_up)
    assert np.isclose(second_milestone_time, unslanted_avg_time_vector[1, 0], atol=0.001)
    assert np.isclose(unslanted_rate_matrix[1,1], -1.0/second_milestone_time, atol=0.001)
    assert np.isclose(unslanted_rate_matrix[0,1], rate_down, atol=0.001)
    assert np.isclose(unslanted_rate_matrix[2,1], rate_up, atol=0.001)
    assert np.isclose(unslanted_transition_matrix[0,1], p_down, atol=0.001)
    assert np.isclose(unslanted_transition_matrix[2,1], p_up, atol=0.001)
    return