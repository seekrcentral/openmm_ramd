"""
Given a single-well system, will construct an energy/diffusion profile,
generate a milestoning model of this, and see how well results are generated
using an approximation of the RAMD simulation and analysis procedure.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp_int

import openmm_ramd.analyze.milestoning as milestoning
import openmm_ramd.analyze.remove_slant as remove_slant
import openmm_ramd.analyze.analyze_log as analyze_log

# Define constants and variables

AVOGADROS_NUMBER = 6.022e23
kT_in_J = 300.0 * 1.380649e-23 # J
kT = kT_in_J * AVOGADROS_NUMBER / 1000 # in kJ / mol
beta = 1.0/(kT)
#print("beta:", beta, "mol/kJ")

D = 1.0 #1.0e2 #7.5e10

# System boundaries
min_z = 0.0
max_z = 10.0

# starting location of particles
start_z = 1.0

# number of finite elements for integration
n_z = 1000
num_milestones = 101
num_xi_bins = 10
start_milestone_index = num_milestones // 10

WELL_DEPTH = 7.0
force_magnitude = 5.0

MODE = 2

def potential_energy_flat(z, force_constant, start=0.0, u=None):
    """
    Returns a flat potential energy profile, but a slant may be optionally added
    by setting 'force_constant' to a number that is not 0.
    """
    if u is None:
        return (z-start)*(-0.0+force_constant)
    else:
        return (z-start)*(-0.0+force_constant)*u
    


def potential_energy_well(z, force_constant, u=None):
    """
    Another single-well potential. This function is used for some other 
    scripts.
    """
    well_depth = 4.0
    minimum = -well_depth
    if u is None:
        y = well_depth*(z-1.0)*(z+1.0)*np.exp(-1.5*z) + z*force_constant
    else:
        y = well_depth*(z-1.0)*(z+1.0)*np.exp(-1.5*z) + z*force_constant*u
    
    return y - minimum

def potential_energy_well_2(z, force_constant, u=None):
    """
    A single-well potential. Used for inputs in this script and some scripts.
    a slant may be optionally added by setting 'force_constant' to a number 
    that is not 0. 'start' indicates the bottom of the well.
    """
    if u is None:
        factor = 1.0
    else:
        factor = u
    
    # Modify these
    well_depth = WELL_DEPTH
    z_offset = 1.0 #1.0
    
    # Don't modify these
    a = -1.0
    b = 0.0
    c = -2.0
    min_z = -np.sqrt(0.5)
    
    displacement = (z-z_offset) + min_z
    min_val = (min_z-a)*(min_z-b)*np.exp(c*min_z)
    #print("min_val:", min_val)
    val = (displacement-a)*(displacement-b)*np.exp(c*displacement)
    norm_val = (-val / min_val) + 1.0
    scaled_val = well_depth * norm_val + displacement*force_constant*factor
    return scaled_val

def diffusion_coefficient_flat(z, D):
    """
    Define a constant diffusion coefficient profile.
    """
    return D
    
def diffusion_coefficient_slanted(z, D):
    """
    Define a diffusion coefficient profile that has a linear slant.
    """
    return D * (1.0 + z / (max_z - min_z))

def make_potential_and_diffusion(z_s, force_constant, potential_energy_func, diffusion_coefficient_func, u=None):
    """
    For each of the finite elements, find the values of the potential
    energies and diffusion coefficients, and return arrays of each.
    """
    potential_energy = np.zeros(z_s.shape)
    min_potential_z = -1.0
    min_potential_y = 1e9
    diffusion_coefficient = np.zeros(z_s.shape)
    for j, z in enumerate(z_s):
        potential_energy[j] = potential_energy_func(z, force_constant, u=u)
        if potential_energy[j] < min_potential_y:
            min_potential_z = z
            min_potential_y = potential_energy[j]
        
        diffusion_coefficient[j] = diffusion_coefficient_func(z, D)
        
    return min_potential_y, potential_energy, diffusion_coefficient

def compute_time(z_s, beta, potential_energy, diffusion_coefficient, h_z, start=2.5):
    """
    Given potential energy & diffusion coefficient arrays, as well as a starting
    position, will compute the expected time that a particle will remain in 
    the system before escaping out the boundary to the right.
    """
    n = len(z_s)
    start_index = int(n*start/(max_z-min_z))
    z_s1 = z_s[:start_index+1]
    n1 = len(z_s1)
    z_s2 = z_s[start_index:]
    n2 = len(z_s2)
    
    stationary_dist = np.zeros(z_s.shape)
    inv_stat_dist = np.zeros(z_s.shape)
    stationary_dist1 = np.zeros(z_s1.shape)
    inv_stat_dist1 = np.zeros(z_s1.shape)
    stationary_dist2 = np.zeros(z_s2.shape)
    inv_stat_dist2 = np.zeros(z_s2.shape)
    
    normalize_const = 1.0
    inv_normalize_const = 1.0 / normalize_const
    
    for j, z in enumerate(z_s):
        stationary_dist[j] = normalize_const * np.exp(-potential_energy[j]*beta)
        inv_stat_dist[j] = inv_normalize_const * np.exp(potential_energy[j]*beta) / diffusion_coefficient[j]
    
    for j1, z1 in enumerate(z_s1):
        stationary_dist1[j1] = normalize_const * np.exp(-potential_energy[j1]*beta)
        inv_stat_dist1[j1] = inv_normalize_const * np.exp(potential_energy[j1]*beta) / diffusion_coefficient[j1]
    
    for j2, z2 in enumerate(z_s2):
        stationary_dist2[j2] = normalize_const * np.exp(-potential_energy[start_index+j2]*beta)
        inv_stat_dist2[j2] = inv_normalize_const * np.exp(potential_energy[start_index+j2]*beta) / diffusion_coefficient[start_index+j2]
    
    # distribution to the left of the source
    time1_integral1_span = np.zeros(z_s1.shape)
    time1_integral2_span = np.zeros(z_s2.shape)
    for j1, z1 in enumerate(z_s1):
        time1_integral1_span[j1] = stationary_dist1[j1]
    
    time_integral1 = sp_int.simps(time1_integral1_span, dx=h_z)
    
    for j2, z2 in enumerate(z_s2):
        time1_integral2_span[j2] = inv_stat_dist2[j2]
        
    
    time_integral2 = sp_int.simps(time1_integral2_span, dx=h_z)
    time1 = time_integral1 * time_integral2
    
    # distribution to the right of the source
    time_integral_span_outer = np.zeros(z_s2.shape)
    for j1, z1 in enumerate(z_s2):
        #if j1 < n2:
        time_integral_span_inner = np.zeros(n2-j1)
        for j2 in range(j1, n2):
            index2 = j2 - j1
            time_integral_span_inner[index2] = inv_stat_dist2[j2]
            
        time_integral_span_outer[j1] = stationary_dist2[j1] * sp_int.simps(time_integral_span_inner, dx=h_z)
        
        #else:
        #    time_integral_span_outer[j1] = 0.0
            
    time2 = sp_int.simps(time_integral_span_outer, dx=h_z)
    time = time1 + time2
    return time, stationary_dist, inv_stat_dist, time_integral_span_outer

def milestoning_statistics_from_profiles(
        milestones, z_s, energy_profile, diffusion_profile, beta):
    base_count = 1000
    num_z_s = len(z_s)
    assert (num_z_s - 1) % (num_milestones - 1) == 0, \
        "num_z_s must be a multiple of num_milestones"
    z_points_per_milestone = (num_z_s - 1) // (num_milestones - 1)
    
    rate_matrix = np.zeros((num_milestones, num_milestones))
    transition_matrix = np.zeros((num_milestones, num_milestones))
    count_matrix = np.zeros((num_milestones, num_milestones))
    avg_time_vector = np.zeros((num_milestones, 1))
    for i, milestone in enumerate(milestones):
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
        
        
        
        i_z = i*z_points_per_milestone
        #b = z_s[i_z]
        if doing_a:
            i_z_minus = (i-1)*z_points_per_milestone
            #a = z_s[i_z_minus]
            z_s_span = z_s[i_z_minus:i_z+1]
            energy_span = energy_profile[i_z_minus:i_z+1]
            diffusion_span = diffusion_profile[i_z_minus:i_z+1]
            #time_b_to_a = get_smoluchowski_time_two_points(
            #    a, b, energy_a, energy_b, diffusion_a, diffusion_b, beta, 
            #    forward=False)
            #time_b_to_a = remove_slant.get_smoluchowski_time(
            #    z_s_span, energy_span, diffusion_span, beta, forward=False)
            #flux_b_to_a = 1 / time_b_to_a
            flux_b_to_a = remove_slant.get_smoluchowski_flux(
                z_s_span, energy_span, diffusion_span, beta, forward=False)
            u_b_to_a = remove_slant.get_smoluchowski_u(
                z_s_span, energy_span, diffusion_span, beta, forward=False)
            total_flux += flux_b_to_a
            total_u += u_b_to_a
            
        else:
            #time_b_to_a = 0.0
            flux_b_to_a = 0.0
            
        if doing_c:
            i_z_plus = (i+1)*z_points_per_milestone
            #c = z_s[i_z_plus]
            z_s_span = z_s[i_z:i_z_plus+1]
            energy_span = energy_profile[i_z:i_z_plus+1]
            diffusion_span = diffusion_profile[i_z:i_z_plus+1]
            #time_b_to_c = get_smoluchowski_time_two_points(
            #    b, c, energy_b, energy_c, diffusion_b, diffusion_c, beta, 
            #    forward=True)
            #time_b_to_c = remove_slant.get_smoluchowski_time(
            #    z_s_span, energy_span, diffusion_span, beta, forward=True)
            #flux_b_to_c = 1 / time_b_to_c
            flux_b_to_c = remove_slant.get_smoluchowski_flux(
                z_s_span, energy_span, diffusion_span, beta, forward=True)
            
            u_b_to_c = remove_slant.get_smoluchowski_u(
                z_s_span, energy_span, diffusion_span, beta, forward=True)
            total_flux += flux_b_to_c
            total_u += u_b_to_c
            
        else:
            #time_b_to_c = 0.0
            flux_b_to_c = 0.0
            
        P_b_to_a = flux_b_to_a / total_flux
        P_b_to_c = flux_b_to_c / total_flux
        #time = P_b_to_a * time_b_to_a + P_b_to_c * time_b_to_c
        time = total_u / total_flux
        
        if doing_a:
            transition_matrix[i-1, i] = P_b_to_a
            rate_matrix[i-1, i] = P_b_to_a / time
            count_matrix[i-1, i] = base_count
        
        if doing_c:
            transition_matrix[i+1, i] = P_b_to_c
            rate_matrix[i+1, i] = P_b_to_c / time
            count_matrix[i+1, i] = base_count
        
        avg_time_vector[i] = time
        rate_matrix[i, i] = -1.0 / time
    return transition_matrix, avg_time_vector, rate_matrix, count_matrix

if __name__ == "__main__":
    # Define a potential energy landscape
    h_z = (max_z - min_z) / n_z
    z_s = np.linspace(min_z, max_z, n_z+1)
    
    potential_energy_func = potential_energy_well_2
    #potential_energy_func = potential_energy_flat
    #diffusion_coefficient_func = diffusion_coefficient_slanted
    diffusion_coefficient_func = diffusion_coefficient_flat

    zeros = np.zeros(z_s.shape)
    min_potential_y1, potential_energy1, diffusion_coefficient1 \
        = make_potential_and_diffusion(z_s, 0.0, potential_energy_func, 
                                       diffusion_coefficient_func)
    #print("min_potential_y1:", min_potential_y1)

    time1, stationary_dist1, inv_stat_dist1, time_integral_span_outer1 \
        = compute_time(z_s, beta, potential_energy1, diffusion_coefficient1, 
                       h_z, start=start_z)
    #print("Expected escape time:", time1, "ps")
    
    """
    # PLOT ORIGINAL FUNCTION
    plt.plot(z_s, potential_energy1, "g", label="Potential energy $W(z) (kJ/mol)$")
    plt.plot(z_s, diffusion_coefficient1, "b", label="Diffusion coefficient $D(z) \\frac{nm^2}{ps}$")
    plt.legend()
    plt.ylabel("$W(z)$ (kJ/mol)")
    plt.xlabel("$z$ (nm)")
    plt.title("Free energy and diffusion profiles")
    plt.ylim((0.0, 2.0*WELL_DEPTH))
    plt.show()
    exit()
    """
    
    stationary_dist_normalized = stationary_dist1 / sp_int.simps(
        stationary_dist1, dx=h_z)
    
    milestones = np.linspace(min_z, max_z, num_milestones)
    milestone_h = milestones[1] - milestones[0]
    bulk_milestone = num_milestones - 1
    
    xi_bins, xi_values, xi_bins_ranges = analyze_log.make_xi_bins(num_xi_bins)
    
    unslanted_energy_profiles = np.zeros((num_xi_bins, num_milestones))
    diffusion_profiles = np.zeros((num_xi_bins, num_milestones))
    counts_profiles = np.zeros((num_xi_bins, num_milestones))
    
    rate_matrix_sum = np.zeros((num_milestones, num_milestones))
    transition_matrix_sum = np.zeros((num_milestones, num_milestones))
    avg_time_vector_sum = np.zeros((num_milestones, 1))
    counts_sum = np.zeros(num_milestones)
    
    for i in range(num_xi_bins):
        xi = xi_values[i]
        slanting_force = xi * force_magnitude
        min_potential_y1, potential_energy_slant, diffusion_coefficient_slant \
        = make_potential_and_diffusion(
            z_s, slanting_force, potential_energy_func, 
            diffusion_coefficient_func)
        
        transition_matrix, avg_time_vector, rate_matrix, count_matrix \
            = milestoning_statistics_from_profiles(
            milestones, z_s, potential_energy_slant, diffusion_coefficient_slant, beta)
        
        if MODE == 1:
            energy_profile, diffusion_profile, counts \
                    = remove_slant.recover_profiles(
                    transition_matrix, avg_time_vector, rate_matrix, count_matrix, 
                    num_milestones, milestone_h, beta, 
                    ref_milestone_index=start_milestone_index)
            
            unslanted_energy_profile = remove_slant.remove_slant(
                milestones, energy_profile, -slanting_force, 
                ref_milestone_index=start_milestone_index)
            
            unslanted_rate_matrix, unslanted_transition_matrix, \
                unslanted_time_vector \
                = remove_slant.milestoning_statistics_from_profiles(
                    milestones, unslanted_energy_profile, diffusion_profile, beta)
        
        elif MODE == 2:
            unslanted_rate_matrix, unslanted_transition_matrix, \
                unslanted_time_vector \
                    = remove_slant.unslant_transition_matrix_constant_approx(
                    milestones, transition_matrix, avg_time_vector, slanting_force,
                    beta)
            counts = milestoning.compute_counts(count_matrix, num_milestones)
        
        for j in range(num_milestones):
            rate_matrix_sum[:, j] += counts[j] * unslanted_rate_matrix[:, j]
            transition_matrix_sum[:, j] += counts[j] * unslanted_transition_matrix[:, j]
            avg_time_vector_sum[j, 0] += counts[j] * unslanted_time_vector[j, 0]
            counts_sum[j] += counts[j]
        
        #unslanted_energy_profiles[i, :] = unslanted_energy_profile
        #diffusion_profiles[i, :] = diffusion_profile
        #counts_profiles[i, :] = counts
        
        recovered_stationary_probs = milestoning.compute_stationary_probabilities(
            unslanted_transition_matrix, unslanted_time_vector, num_milestones)
        
        recovered_energy_profile = remove_slant.make_energy_profile(
            recovered_stationary_probs, beta, start_milestone_index)
        
        #stationary_probs_norm = stationary_probs / sp_int.simps(stationary_probs, dx=milestone_h)
        
        # PLOT SLANTED_PROFILES
        #plt.plot(milestones, energy_profile, label=f"$\\xi$={xi:.2f}")
        
        # PLOT UNSLANTED PROFILES
        #plt.plot(milestones, unslanted_energy_profile, label=f"$\\xi$={xi:.2f}")
        
        # PLOT RECOVERED PROFILES
        #plt.plot(milestones, recovered_energy_profile, label=f"$\\xi$={xi:.2f}")
    
    #plt.plot(z_s, potential_energy1, "g", label="Potential energy $W(z) (kJ/mol)$")
    #plt.plot(milestones, energy_profile, "r", label="Recovered Potential energy $W(z) (kJ/mol)$")
    #plt.plot(z_s, diffusion_coefficient1, "b", label="Diffusion coefficient $D(z) \\frac{nm^2}{ps}$")
    #plt.plot(milestones, diffusion_profile, "k", label="Recovered Diffusion coefficient $D(z) \\frac{nm^2}{ps}$")
    
    rate_matrix_avg = np.zeros((num_milestones, num_milestones))
    transition_matrix_avg = np.zeros((num_milestones, num_milestones))
    avg_time_vector_avg = np.zeros((num_milestones, 1))
        
    for j in range(num_milestones):
        rate_matrix_avg[:, j] = rate_matrix_sum[:, j] / counts_sum[j]
        transition_matrix_avg[:, j] = transition_matrix_sum[:, j] / counts_sum[j]
        avg_time_vector_avg[j, 0] = avg_time_vector_sum[j, 0] / counts_sum[j]
    
    averaged_stationary_probs = milestoning.compute_stationary_probabilities(
        transition_matrix_avg, avg_time_vector_avg, num_milestones)
    
    averaged_energy_profile = remove_slant.make_energy_profile(
        averaged_stationary_probs, beta, start_milestone_index)
        
    # PLOT AVG'D ENERGY PROFILE
    
    #energy_profile_avg, diffusion_profile_avg \
    #    = remove_slant.average_unslanted_profiles(
    #    num_xi_bins, num_milestones, beta, unslanted_energy_profiles,
    #    diffusion_profiles, counts_profiles)
    
    #time_estimate = analyze_log.compute_time(
    #    milestones, energy_profile_avg, diffusion_profile_avg, beta, 
    #    start_milestone_index, bulk_milestone)
    
    time_estimate = analyze_log.compute_time(rate_matrix_avg, start_milestone_index, bulk_milestone)
    
    print("recovered time:", time_estimate, "ps")
    print("original time:", time1, "ps")
    
    plt.plot(milestones, averaged_energy_profile, "r", label="Recovered Potential energy $W(z) (kJ/mol)$")
    #plt.plot(milestones, diffusion_profile_avg, "k", label="Recovered Diffusion coefficient $D(z) \\frac{nm^2}{ps}$")
    #plt.plot(z_s, potential_energy1, "g", label="Potential energy $W(z) (kJ/mol)$")
    #plt.plot(z_s, diffusion_coefficient1, "b", label="Diffusion coefficient $D(z) \\frac{nm^2}{ps}$")
    
    #plt.plot(z_s, stationary_dist_normalized, "g")
    #plt.plot(milestones, stationary_probs_norm, "r")
    plt.legend()
    plt.ylabel("$W(z)$ (kJ/mol)")
    plt.xlabel("$z$ (nm)")
    plt.title("Free energy and diffusion profiles")
    plt.ylim((0.0, 2.0*WELL_DEPTH))
    plt.show()