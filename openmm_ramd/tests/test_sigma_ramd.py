"""
Test openmm_ramd sigma_ramd.py functions.
"""

import numpy as np
import scipy.integrate as sp_int
import matplotlib.pyplot as plt

import utils
import openmm_ramd.analyze.milestoning as milestoning
import openmm_ramd.analyze.sigma_ramd as sigma_ramd

# TODO: understand why we have to use_abs=True in this case...

def make_test_xi_bin_time_profiles(num_xi_bins, milestones, forceRAMD, D, beta):
    xi_a = -1.0
    xi_b = 1.0
    xi_span = xi_b - xi_a
    xi_s_boundaries = np.linspace(xi_a, xi_b, num_xi_bins+1)
    xi_s = np.zeros(num_xi_bins)
    for i in range(num_xi_bins):
        xi_s[i] = 0.5*(xi_s_boundaries[i] + xi_s_boundaries[i+1])
        
    h_u = xi_span / (num_xi_bins)
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

def test_functional_expansion_1d_ramd_make_zeroth_order_terms():
    beta=1.0 # TODO: fill out
    D = 1.0
    min_location = 0.0
    max_location = 10.0
    starting_cv_val = 0.0
    forceRAMD1 = 0.1
    forceRAMD2 = 1.0
    num_xi_bins = 41
    num_milestones = 11
    milestones = np.linspace(min_location, max_location, num_milestones)
    xi_bin_time_profiles1 = make_test_xi_bin_time_profiles(
        num_xi_bins, milestones, forceRAMD1, D, beta)
    
    xi_bin_times1 = np.zeros(num_xi_bins)
    for i in range(num_xi_bins):
        xi_bin_times1[i] = xi_bin_time_profiles1[i, -1, 0]
    
    calc1 = sigma_ramd.functional_expansion_1d_ramd(
        xi_bin_time_profiles1, xi_bin_times1, force_constant=forceRAMD1, beta=beta,
        min_cv_val=min_location, max_cv_val=max_location, 
        starting_cv_val=starting_cv_val)
    calc1.make_zeroth_order_terms()
    
    assert np.isclose(calc1.T, 49.54, atol=0.1)
    #print("calc1.T:", calc1.T)
    for i in range(num_xi_bins):
        xi = calc1.xi_s[i]
        this_time = utils.slant_time(
            max_location, min_location, D, beta, forceRAMD1*xi, use_abs=True)
        this_flux = 1.0 / this_time
        assert np.isclose(calc1.flux_array[i], this_flux, atol=0.01)
    
    xi_bin_time_profiles2 = make_test_xi_bin_time_profiles(
        num_xi_bins, milestones, forceRAMD2, D, beta)
    
    xi_bin_times2 = np.zeros(num_xi_bins)
    for i in range(num_xi_bins):
        xi_bin_times2[i] = xi_bin_time_profiles2[i, -1, 0]
    
    calc2 = sigma_ramd.functional_expansion_1d_ramd(
        xi_bin_time_profiles2, xi_bin_times2, force_constant=forceRAMD1, beta=beta,
        min_cv_val=min_location, max_cv_val=max_location, 
        starting_cv_val=starting_cv_val)
    calc2.make_zeroth_order_terms()
    
    assert np.isclose(calc2.T, 29.49, atol=0.1)
    #print("calc2.T:", calc2.T)
    
    for i in range(num_xi_bins):
        xi = calc2.xi_s[i]
        this_time = utils.slant_time(
            max_location, min_location, D, beta, forceRAMD2*xi, use_abs=True)
        this_flux = 1.0 / this_time
        #print("i:", i, "xi:", xi, "this_flux:", this_flux, 
        #      "calc2.flux_array[i]:", calc2.flux_array[i])
        assert np.isclose(calc2.flux_array[i], this_flux, atol=0.01)

def test_functional_expansion_1d_ramd_make_first_order_terms():
    beta=1.0 # TODO: fill out
    D = 1.0
    min_location = 0.0
    max_location = 10.0
    starting_cv_val = 0.0
    forceRAMD1 = 0.1
    num_xi_bins = 40
    num_milestones = 11
    xi_span = 2.0
    h_xi = xi_span / (num_xi_bins)
    h_z = (max_location - min_location) / (num_milestones - 1)
    
    milestones = np.linspace(min_location, max_location, num_milestones)
    xi_bin_time_profiles1 = make_test_xi_bin_time_profiles(
        num_xi_bins, milestones, forceRAMD1, D, beta)
    
    xi_bin_times1 = np.zeros(num_xi_bins)
    for i in range(num_xi_bins):
        xi_bin_times1[i] = xi_bin_time_profiles1[i, -1, 0]
    
    calc1 = sigma_ramd.functional_expansion_1d_ramd(
        xi_bin_time_profiles1, xi_bin_times1, force_constant=forceRAMD1, beta=beta,
        min_cv_val=min_location, max_cv_val=max_location, 
        starting_cv_val=starting_cv_val)
    calc1.make_zeroth_order_terms()
    calc1.make_first_order_terms()
    calc1.make_second_order_terms()
    print("calc1.second_order_correction:", calc1.second_order_correction)
    
    # Now explicitly compute the term
    exact_fluxes = np.zeros(num_xi_bins)
    exact_fluxes_sq = np.zeros(num_xi_bins)
    for i in range(num_xi_bins):
        xi = calc1.xi_s[i]
        this_time = utils.slant_time(
            max_location, min_location, D, beta, forceRAMD1*xi, use_abs=True)
        this_flux = 1.0 / this_time
        exact_fluxes[i] = this_flux
        exact_fluxes_sq[i] = this_flux**2
        
    exact_T = xi_span / sigma_ramd.midpoint_Riemann(exact_fluxes, dx=h_xi)
    #print("exact_T:", exact_T)
    
    #integrated_flux_sq = xi_span / sigma_ramd.midpoint_Riemann(exact_fluxes_sq, dx=h_xi)
    
    first_term_points1 = np.zeros(num_xi_bins)
    first_term_points2 = np.zeros(num_xi_bins)
    
    for i in range(num_xi_bins):
        xi = calc1.xi_s[i]
        # This needs to be put inside a xi integral
        #forceRAMD1 = 0.0
        time_derivatives1 = np.zeros(num_milestones)
        time_derivatives2 = np.zeros(num_milestones)
        for j in range(num_milestones):
            num_inside_pts1 = j+1
            num_inside_pts2 = num_milestones - j
            z = min_location + j * h_z
            
            if num_inside_pts1 > 1:
                inside_integral1 = np.zeros(num_inside_pts1)
                for k in range(num_inside_pts1):
                    z_prime = min_location + k * h_z
                    inside_integral1[k] = np.exp(-beta * forceRAMD1*xi * z_prime)
                    #print("z:", z, "z_prime:", z_prime)
                
                time_deriv1 = np.exp(beta * forceRAMD1*xi * z) / D \
                * sp_int.simps(inside_integral1, dx=h_z)
            
            else:
                time_deriv1 = 0.0
            
            if num_inside_pts2 > 1:
                inside_integral2 = np.zeros(num_inside_pts2)
                for k in range(num_inside_pts2):
                    z_prime = z + k * h_z
                    inside_integral2[k] = np.exp(beta * forceRAMD1*xi * z_prime) / D
                    #print("z:", z, "z_prime:", z_prime, "forceRAMD1*xi:", forceRAMD1*xi)
                    
                time_deriv2 = np.exp(-beta * forceRAMD1*xi * z) \
                    * sp_int.simps(inside_integral2, dx=h_z)
                    
            else:
                time_deriv2 = 0.0
                
            time_derivatives1[j] = z * time_deriv1
            time_derivatives2[j] = z * time_deriv2
            
        first_time_deriv1 = sp_int.simps(time_derivatives1, dx=h_z)
        first_time_deriv2 = sp_int.simps(time_derivatives2, dx=h_z)
        
        #print("i:", i, "xi:", xi, "exact_fluxes_sq[i]:", exact_fluxes_sq[i], 
        #      "first_time_deriv:", first_time_deriv)
        first_term_points1[i] = exact_fluxes_sq[i] * first_time_deriv1 * xi
        first_term_points2[i] = exact_fluxes_sq[i] * first_time_deriv2 * xi
        #print("i:", i, "xi:", xi, "first_term_points1[i]:", first_term_points1[i])
    
    
    #plt.plot(calc1.xi_s, np.linspace(first_term_points1[0], first_term_points1[-1], num_xi_bins), "k")
    #plt.plot(calc1.xi_s, np.linspace(first_term_points2[0], first_term_points2[-1], num_xi_bins), "k")
    #plt.plot(calc1.xi_s, first_term_points1, "r")
    #plt.plot(calc1.xi_s, first_term_points2, "g")
    #plt.show()
    
    
    #print("exact_T**2:", exact_T**2)
    #print("xi_span:", xi_span)
    integral1 = sigma_ramd.midpoint_Riemann(first_term_points1, dx=h_xi)
    integral2 = sigma_ramd.midpoint_Riemann(first_term_points2, dx=h_xi)
    #print("integral1:", integral1)
    #print("integral2:", integral2)
    correction_1st_order = exact_T**2 \
        * integral1 / xi_span
    #print("correction_1st_order:", correction_1st_order)
    exponent1 = beta * forceRAMD1 * correction_1st_order
    first_term = np.exp(exponent1)
    #print("exact exponent1:", exponent1)
    exact_first_order_correction = exact_T * first_term
    #print("exact_first_order_correction:", exact_first_order_correction)
    
def test_functional_expansion_1d_ramd_time_estimate_pade_approx_second_order():
    beta = 1.0 # TODO: fill out
    D = 1.0
    min_location = 0.0
    max_location = 10.0
    starting_cv_val = 0.0
    forceRAMD1 = 0.1
    num_xi_bins = 40
    num_milestones = 11
    xi_span = 2.0
    h_xi = xi_span / (num_xi_bins)
    h_z = (max_location - min_location) / (num_milestones - 1)
    
    milestones = np.linspace(min_location, max_location, num_milestones)
    xi_bin_time_profiles1 = make_test_xi_bin_time_profiles(
        num_xi_bins, milestones, forceRAMD1, D, beta)
    
    xi_bin_times1 = np.zeros(num_xi_bins)
    for i in range(num_xi_bins):
        xi_bin_times1[i] = xi_bin_time_profiles1[i, -1, 0]
    
    calc1 = sigma_ramd.functional_expansion_1d_ramd(
        xi_bin_time_profiles1, xi_bin_times1, force_constant=forceRAMD1, beta=beta,
        min_cv_val=min_location, max_cv_val=max_location, 
        starting_cv_val=starting_cv_val)
    time_estimate_pade = calc1.time_estimate_pade_approx_second_order()
    print("calc1.T:", calc1.T)
    print("time_estimate_pade:", time_estimate_pade)
    