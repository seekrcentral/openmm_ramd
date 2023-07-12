"""
Run a sigma-RAMD calculation on the time profile data generated from
a RAMD log file, or perhaps some other means.
"""

import numpy as np
import scipy.integrate as sp_int
import matplotlib.pyplot as plt

def midpoint_Riemann(data, dx):
    return dx * np.sum(data)

class functional_expansion_1d_ramd():
    def __init__(self, xi_bin_time_profiles, xi_bin_times, force_constant, beta, 
                 min_cv_val, max_cv_val, starting_cv_val):
        self.xi_bin_time_profiles = xi_bin_time_profiles
        [num_xi_bins, num_milestones, dummy] = xi_bin_time_profiles.shape
        self.xi_bin_times = xi_bin_times
        #self.num_milestones = num_milestones
        self.n = num_milestones
        self.n_xi = num_xi_bins
        self.A = force_constant
        self.beta = beta
        self.a = min_cv_val
        self.b = starting_cv_val
        self.c = max_cv_val
        self.h_z = (self.c - self.a) / (self.n-1)
        self.xi_a = -1.0
        self.xi_b = 1.0
        self.xi_span = self.xi_b - self.xi_a
        self.h_xi = self.xi_span / (self.n_xi)
        self.z_s = np.linspace(self.a, self.c, self.n)
        xi_bins = np.linspace(self.xi_a, self.xi_b, self.n_xi+1)
        self.xi_s = np.zeros(self.n_xi)
        for i in range(self.n_xi):
            self.xi_s[i] = 0.5*(xi_bins[i] + xi_bins[i+1])
        return
    
    
    def make_zeroth_order_terms(self):
        # inverse times for each slice of xi
        self.flux_array = np.zeros(self.n_xi)
        self.time_integrals_xi1 = None
        start_index = int((self.b - self.a)/self.h_z)
        #print("start_index:", start_index)
        for i, xi in enumerate(self.xi_s):
            time_start_to_finish = self.xi_bin_times[i]
            #print("i:", i, "time_start_to_finish:", time_start_to_finish)
            self.flux_array[i] = 1.0 / time_start_to_finish
            
        print("self.flux_array:", self.flux_array)
        #self.T = self.xi_span / sp_int.simps(self.flux_array, dx=self.h_xi)
        self.T = self.xi_span / midpoint_Riemann(self.flux_array, dx=self.h_xi)
        print("T:", self.T)
        
    def make_first_order_terms(self):
        boundary_terms = np.zeros(self.n_xi)
        integrated_times_b_to_z = np.zeros(self.n_xi)
        integrated_times_z_to_c = np.zeros(self.n_xi)
        self.time_integrals_xi1 = np.zeros(self.n_xi)
        for i, xi in enumerate(self.xi_s):
            time_span_b_to_z = np.zeros(self.n)
            for j, z in enumerate(self.z_s):
                time_span_b_to_z[j] = self.xi_bin_time_profiles[i, j, 0]
            integrated_time_b_to_z = sp_int.simps(time_span_b_to_z, dx=self.h_z)
            
            boundary_term1 = (self.c-self.a)*self.xi_bin_time_profiles[i, -1, 0]
            # TODO: check what is valid to do here for a source a != b
            #boundary_term1 = (self.c-self.a)*self.xi_bin_times[i]
            #boundary_term1 = (self.c-self.b)*self.xi_bin_times[i]
            
            time_span_z_to_c = np.zeros(self.n)
            for j, z in enumerate(self.z_s):
                time_span_z_to_c[j] = self.xi_bin_time_profiles[i, -1, j]
            integrated_time_z_to_c = sp_int.simps(time_span_z_to_c, dx=self.h_z)
            
            time_value \
                = (boundary_term1 - integrated_time_b_to_z \
                   - integrated_time_z_to_c) * self.flux_array[i]**2 * xi
            boundary_terms[i] = boundary_term1
            integrated_times_b_to_z[i] = integrated_time_b_to_z
            integrated_times_z_to_c[i] = integrated_time_z_to_c
            self.time_integrals_xi1[i] = time_value
            if np.isfinite(time_value):
                self.time_integrals_xi1[i] = time_value
            else:
                self.time_integrals_xi1[i] = 0.0
            
        #print("boundary_terms:", boundary_terms)
        #plt.plot(self.xi_s, boundary_terms)
        #plt.ylim((-0.0, 400.0))
        #plt.show()
        
        time_integral = midpoint_Riemann(
            self.time_integrals_xi1, dx=self.h_xi) \
            * self.T**2 / self.xi_span
        #print("time_integral:", time_integral)
        #print("self.beta:", self.beta)
        #print("self.A:", self.A)
        self.exponent1 = self.beta*self.A*time_integral
        #print("self.exponent1:", self.exponent1)
        self.first_order_correction = self.T * np.exp(self.exponent1/self.T)
        print("self.first_order_correction:", self.first_order_correction)
        
        return
    
    def make_second_order_terms(self):
        time_integrals_xi = np.zeros(self.n_xi)
        for i, xi in enumerate(self.xi_s):
            # Test the on-diagonal b to z
            time_span_on_diag_b_to_z = np.zeros(self.n)
            for j1, z in enumerate(self.z_s):
                integral_span_inner1 = np.zeros(j1+1)
                for j2 in range(j1+1):
                   
                    integral_span_inner1[j2] \
                        = self.xi_bin_time_profiles[i, j2, 0]
                
                time_b_to_z = sp_int.simps(integral_span_inner1, dx=self.h_z)
                time_span_on_diag_b_to_z[j1] = time_b_to_z
            
            integrated_time_b_to_z = 2.0 \
                * sp_int.simps(time_span_on_diag_b_to_z, dx=self.h_z)
            
            # boundary term c**2
            boundary_term_b_to_c1 = (self.c-self.a)**2 \
                * self.xi_bin_time_profiles[i, -1, 0]
            
            # boundary term 2*c
            integral_span_inner1 = np.zeros(self.n)
            for j, z in enumerate(self.z_s):
                integral_span_inner1[j] = self.xi_bin_time_profiles[i, j, 0]
            boundary_term_b_to_c2 = 2.0 * (self.c-self.a) \
                * sp_int.simps(integral_span_inner1, dx=self.h_z)
            on_diag_b_to_z = boundary_term_b_to_c1 - boundary_term_b_to_c2 \
                + integrated_time_b_to_z
            
            # on diagonal z to c
            time_span_z_to_c = np.zeros(self.n)
            for j1, z in enumerate(self.z_s):
                integral_span_inner1 = np.zeros(self.n-j1)
                for j2 in range(j1, self.n):
                    index2 = self.n-j2-1
                    
                    integral_span_inner1[index2] = self.xi_bin_time_profiles[i, -1, j2]
                
                time_z_to_c = sp_int.simps(integral_span_inner1, dx=self.h_z)
                time_span_z_to_c[j1] = time_z_to_c
            
            integrated_time_z_to_c = 2.0 \
                * sp_int.simps(time_span_z_to_c, dx=self.h_z)
            on_diag_est = on_diag_b_to_z + integrated_time_z_to_c
            
            # Test the off-diagonal b to z
            
            # boundary term b to z
            integral_span_inner1 = np.zeros(self.n)
            for j1, z in enumerate(self.z_s):
                integral_span_inner1[j1] = self.xi_bin_time_profiles[i, -1, j1]
            off_diag_b_to_z_boundary_term = self.c \
                * sp_int.simps(integral_span_inner1, dx=self.h_z)
            
            time_span_zprime_to_z = np.zeros(self.n)
            for j1, z in enumerate(self.z_s):
                integral_span_inner1 = np.zeros(j1+1)
                for j2 in range(j1+1):
                    integral_span_inner1[j2] \
                        = self.xi_bin_time_profiles[i, j1, j2]
                
                time_b_to_z = sp_int.simps(integral_span_inner1, dx=self.h_z)
                time_span_zprime_to_z[j1] = time_b_to_z
            
            off_diag_zprime_to_z = sp_int.simps(
                time_span_zprime_to_z, dx=self.h_z)
            
            time_span_z_to_zprime = np.zeros(self.n)
            for j1, z in enumerate(self.z_s):
                integral_span_inner1 = np.zeros(self.n-j1)
                for j2 in range(j1, self.n):
                    index2 = j2 - j1
                    integral_span_inner1[index2] \
                        = self.xi_bin_time_profiles[i, j2, j1]
                
                time_b_to_z = sp_int.simps(integral_span_inner1, dx=self.h_z)
                time_span_z_to_zprime[j1] = time_b_to_z
            
            off_diag_z_to_zprime = sp_int.simps(
                time_span_z_to_zprime, dx=self.h_z)
            
            off_diag_est = 2.0 * off_diag_b_to_z_boundary_term \
                - off_diag_zprime_to_z - off_diag_z_to_zprime
            
            #print("xi:", xi, "on_diag_est:", on_diag_est, "off_diag_est:", off_diag_est)
            #print("xi:", xi, "off_diag_b_to_z_boundary_term:", off_diag_b_to_z_boundary_term, 
            #                 "off_diag_zprime_to_z:", off_diag_zprime_to_z,
            #                 "off_diag_z_to_zprime:", off_diag_z_to_zprime)
            
            time_value = (-on_diag_est + off_diag_est) \
                * self.flux_array[i]**2 * xi**2
            if np.isfinite(time_value):
                time_integrals_xi[i] = time_value
            else:
                time_integrals_xi[i] = 0.0
            #print("xi:", xi, "time_integrals_xi[i]:", time_integrals_xi[i])
        
        # Down here, integrate over u values
        time_integral = -midpoint_Riemann(time_integrals_xi, dx=self.h_xi) \
            * self.T**2 / self.xi_span
        time_integrals_xi1_flux3 = np.zeros(self.n_xi)
        
        #plt.plot(self.xi_s, time_integrals_xi)
        #plt.ylim((-4000.0, 4000.0))
        #plt.show()
        
        term1 = 2 * midpoint_Riemann(self.time_integrals_xi1, dx=self.h_xi)**2 \
            * self.T**3 / self.xi_span**2
        
        for i, xi in enumerate(self.xi_s):
            time_value = self.time_integrals_xi1[i]**2 \
                / self.flux_array[i]
            if np.isfinite(time_value):
                time_integrals_xi1_flux3[i] = time_value
            else:
                time_integrals_xi1_flux3[i] = 0.0
            
        term2 = -2 * midpoint_Riemann(time_integrals_xi1_flux3, dx=self.h_xi) \
            * self.T**2 / self.xi_span
        
        term3 = time_integral
        #print("term1:", term1)
        #print("term2:", term2)
        #print("term3:", term3)
        term_sum = term1 + term2 + term3
        exponent2_a = self.beta**2*self.A**2*term_sum \
            / self.T
        exponent2_b = (self.exponent1 / self.T)**2
        self.exponent2 = (exponent2_a - exponent2_b)/2.0
        self.second_order_correction = self.first_order_correction \
            * np.exp(self.exponent2)
        #self.second_order_correction = self.beta**2*self.A**2*time_integral/2.0
    
    # Truncated Taylor series
    def time_estimate_taylor_series_second_order(self):
        self.make_zeroth_order_terms()
        self.make_first_order_terms()
        self.make_second_order_terms()
        time_estimate = self.second_order_correction
        return time_estimate
    
    def time_estimate_pade_approx_second_order(self):
        """
        I'M DOING THIS WRONG: does there need to be an accounting for
        the independent variable (the force constant)?
        """
        self.make_zeroth_order_terms()
        self.make_first_order_terms()
        self.make_second_order_terms()
        a_0 = np.log(self.T)
        a_1 = self.exponent1
        a_2 = self.exponent2
        time_estimate = np.exp((a_0+(a_1-(a_2*a_0/a_1)))/(1.0-(a_2/a_1)))
        return time_estimate