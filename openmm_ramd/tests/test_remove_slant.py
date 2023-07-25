"""
test_remove_slant.py
"""

import numpy as np
import matplotlib.pyplot as plt

import utils
import openmm_ramd.analyze.remove_slant as remove_slant
import openmm_ramd.analyze.analyze_log as analyze_log

def test_get_smoluchowski_time():
    a = 0.0
    b = 3.0
    energy_a = 0.0
    energy_b = 0.0
    diffusion_a = 1.0
    diffusion_b = 1.0
    beta = 1.0
    flat_time_forward = remove_slant.get_smoluchowski_time_two_points(
        a, b, energy_a, energy_b, diffusion_a, diffusion_b,
                          beta, forward=True)
    assert np.isclose(flat_time_forward, 4.5)
    
    flat_time_backward = remove_slant.get_smoluchowski_time_two_points(
        a, b, energy_a, energy_b, diffusion_a, diffusion_b,
                          beta, forward=False)
    assert np.isclose(flat_time_backward, 4.5)
    
    energy_b = 2.0
    force_constant = (energy_b - energy_a) / (b - a)
    correct_time = utils.slant_time(a, b, diffusion_a, beta, force_constant)
    smol_time = remove_slant.get_smoluchowski_time_two_points(
        a, b, energy_a, energy_b, diffusion_a, diffusion_b,
                          beta, forward=True)
    assert np.isclose(correct_time, smol_time, atol=1e-3)
    
    energy_a = 2.0
    energy_b = 0.0
    force_constant = (energy_b - energy_a) / (b - a)
    smol_time = remove_slant.get_smoluchowski_time_two_points(
        a, b, energy_a, energy_b, diffusion_a, diffusion_b,
                          beta, forward=False)
    assert np.isclose(correct_time, smol_time, atol=1e-3)
    return

def test_milestoning_statistics_from_profiles():
    num_points = 11
    milestones = np.linspace(0.0, 10.0, num_points)
    energy_profile = np.zeros(num_points)
    diffusion_profile = np.ones(num_points)
    beta = 1.0
    rate_matrix = remove_slant.milestoning_statistics_from_profiles(
        milestones, energy_profile, diffusion_profile, beta)
    
    return

def test_recover_profiles_slant_system():
    num_milestones = 11
    h = 1.0
    D = 1.0
    A = -1.0
    beta = 1.0
    base_count = 1000
    milestones = np.linspace(0.0, h*(num_milestones-1), num_milestones)
    transition_matrix, time_vector, time_vector_up, time_vector_down, \
        rate_matrix, count_matrix = utils.slant_milestoning_model(
            num_milestones, A, h, D, base_count)
    
    energy_profile, diffusion_profile, counts = remove_slant.recover_profiles(
        transition_matrix, time_vector, rate_matrix, count_matrix, 
        num_milestones, h, beta)
    
    unslanted_energy_profile = remove_slant.remove_slant(
            milestones, energy_profile, -A)
    
    #plt.plot(milestones, energy_profile)
    #plt.plot(milestones, diffusion_profile)
    #plt.plot(milestones, unslanted_energy_profile)
    #plt.show()
    return
    
def test_average_unslanted_profiles_slant_system():
    num_milestones = 11
    num_xi_bins = 10
    h = 1.0
    D = 1.0
    A = 1.0
    beta = 1.0
    base_count = 1000
    milestones = np.linspace(0.0, h*(num_milestones-1), num_milestones)
    xi_bins, xi_values, xi_bins_ranges = analyze_log.make_xi_bins(num_xi_bins)
    
    unslanted_energy_profiles = np.zeros((num_xi_bins, num_milestones))
    diffusion_profiles = np.zeros((num_xi_bins, num_milestones))
    counts_profiles = np.zeros((num_xi_bins, num_milestones))
    
    for i in range(num_xi_bins):
        xi = xi_values[i]
        slanting_force = xi * A
        transition_matrix, time_vector, time_vector_up, time_vector_down, \
            rate_matrix, count_matrix = utils.slant_milestoning_model(
            num_milestones, slanting_force, h, D, base_count)
            
        energy_profile, diffusion_profile, counts \
            = remove_slant.recover_profiles(
            transition_matrix, time_vector, rate_matrix, count_matrix, 
            num_milestones, h, beta)
        unslanted_energy_profile = remove_slant.remove_slant(
            milestones, energy_profile, -slanting_force)
        
        unslanted_energy_profiles[i, :] = unslanted_energy_profile
        diffusion_profiles[i, :] = diffusion_profile
        counts_profiles[i, :] = counts
        
        #plt.plot(milestones, unslanted_energy_profile, label=f"{xi:.2f}")
    
    #plt.legend()
    #plt.show()
    
    energy_profile_avg, diffusion_profile_avg \
        = remove_slant.average_unslanted_profiles(
        num_xi_bins, num_milestones, beta, unslanted_energy_profiles,
        diffusion_profiles, counts_profiles)
    
    #plt.plot(milestones, energy_profile_avg)
    #plt.plot(milestones, diffusion_profile_avg)
    #plt.show()
    return