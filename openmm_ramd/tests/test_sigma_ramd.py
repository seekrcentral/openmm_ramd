"""
Test openmm_ramd sigma_ramd.py functions.
"""

import numpy as np

import utils
import openmm_ramd.analyze.milestoning as milestoning
import openmm_ramd.analyze.sigma_ramd as sigma_ramd

# TODO: understand why we have to use_abs=True in this case...

def make_test_xi_bin_time_profiles(num_xi_bins, milestones, forceRAMD, D, beta):
    xi_a = -1.0
    xi_b = 1.0
    xi_span = xi_b - xi_a
    xi_s = np.linspace(xi_a, xi_b, num_xi_bins)
    h_u = xi_span / (num_xi_bins-1)
    num_milestones = len(milestones)
    xi_bin_time_profiles = np.zeros(
        (num_xi_bins, num_milestones, num_milestones))
    for i, xi in enumerate(xi_s):
        force_constant = xi * forceRAMD
        for j, milestone1 in enumerate(milestones):
            #z1 = milestone1
            for k, milestone2 in enumerate(milestones):
                #z2 = milestone2
                xi_bin_time_profiles[i,k,j] = utils.slant_time(
                    milestone2, milestone1, D, beta, force_constant, 
                    use_abs=True)
                #if np.isclose(xi, -0.2):
                #    print("time", milestone1, "to", milestone2, ":", xi_bin_time_profiles[i,j,k])
                
    return xi_bin_time_profiles

def test_linear_system():
    beta=1.0 # TODO: fill out
    D = 1.0
    min_location = 0.0
    max_location = 10.0
    starting_cv_val = 0.0
    forceRAMD = 0.1
    num_xi_bins = 11
    num_milestones = 11
    milestones = np.linspace(min_location, max_location, num_milestones)
    xi_bin_time_profiles = make_test_xi_bin_time_profiles(
        num_xi_bins, milestones, forceRAMD, D, beta)
    
    xi_bin_times = np.zeros(num_xi_bins)
    for i in range(num_xi_bins):
        xi_bin_times[i] = xi_bin_time_profiles[i, -1, 0]
    
    calc = sigma_ramd.functional_expansion_1d_ramd(
        xi_bin_time_profiles, xi_bin_times, force_constant=forceRAMD, beta=beta,
        min_cv_val=min_location, max_cv_val=max_location, 
        starting_cv_val=starting_cv_val)
    calc.make_zeroth_order_terms()
    # Value came from script ramd_flat_to_slant_test.py
    assert np.isclose(calc.T, 49.54, atol=0.1)
    print("calc.T:", calc.T)
    #calc.make_first_order_terms()
    #calc.make_second_order_terms()
    #print("calc.second_order_correction", calc.second_order_correction)
    time_estimate = calc.time_estimate_taylor_series_second_order()
    print("time_estimate_taylor_series_second_order:", time_estimate)
    assert np.isclose(time_estimate, 50.0, atol=0.1)
    return