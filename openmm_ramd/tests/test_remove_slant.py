"""
test_remove_slant.py
"""

import numpy as np
import scipy.integrate as sp_int
import matplotlib.pyplot as plt

import openmm_ramd.analyze.remove_slant as remove_slant

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

def test_unslant_transition_matrix_from_energy_profile():
    num_milestones = 11
    milestones = np.linspace(0.0, 10.0, num_milestones)
    slanting_force = 0.0
    energy_profile = np.linspace(0.0, 20.0, num_milestones)
    diffusion_profile = np.ones(num_milestones)
    beta = 0.5
    
    unslanted_rate_matrix, unslanted_transition_matrix, \
        unslanted_avg_time_vector, unslanted_energy_profile \
        = remove_slant.unslant_transition_matrix_from_energy_profile(
        milestones, energy_profile, diffusion_profile, slanting_force, beta)
    
    b = 0.0
    c = 1.0
    energy_b = 0.0
    energy_c = 2.0
    diffusion = 1.0
    A = (energy_c - energy_b) / (c - b)
    
    first_milestone_flux = beta * A * diffusion / (np.exp(beta*A*(c-b)) - 1.0)
    first_milestone_int_u = ((np.exp(beta*A*(c-b)) - 1.0)/(beta*A) - c + b) / (np.exp(beta*A*(c-b)) - 1.0)
    first_milestone_time = ((np.exp(beta*A*(c-b)) - 1.0)/(beta*A) - c + b) / (beta*A*diffusion)
    
    assert np.isclose(first_milestone_time, unslanted_avg_time_vector[0, 0])   
    assert np.isclose(unslanted_rate_matrix[0,0], -1.0/first_milestone_time)
    assert np.isclose(unslanted_rate_matrix[1,0], 1.0/first_milestone_time)
    assert unslanted_transition_matrix[1,0] == 1.0
        
    factor1 = 1.0 / (1.0 - np.exp(-beta*A*(c-b)))
    second_milestone_flux1 = beta * A * diffusion * factor1
    second_milestone_int_u1 = factor1 * (c - b - (1.0 - np.exp(-beta*A*(c-b)))/(beta*A))
    
    factor2 = 1.0 / (np.exp(beta*A*(c-b)) - 1.0)
    second_milestone_flux2 = beta * A * diffusion * factor2
    second_milestone_int_u2 = factor2 * ((np.exp(beta*A*(c-b)) - 1.0)/(beta*A) - c + b)
    
    second_milestone_time = (second_milestone_int_u1 + second_milestone_int_u2) \
        / (second_milestone_flux1 + second_milestone_flux2)
    
    p_down = second_milestone_flux1 / (second_milestone_flux1 + second_milestone_flux2)
    p_up = second_milestone_flux2 / (second_milestone_flux1 + second_milestone_flux2)
    rate_down = p_down/second_milestone_time
    rate_up = p_up/second_milestone_time
    
    assert np.isclose(second_milestone_time, unslanted_avg_time_vector[1, 0])
    assert np.isclose(unslanted_rate_matrix[1,1], -1.0/second_milestone_time)
    assert np.isclose(unslanted_rate_matrix[0,1], p_down/second_milestone_time)
    assert np.isclose(unslanted_rate_matrix[2,1], p_up/second_milestone_time)
    assert np.isclose(unslanted_transition_matrix[0,1], p_down)
    assert np.isclose(unslanted_transition_matrix[2,1], p_up)
    return
    