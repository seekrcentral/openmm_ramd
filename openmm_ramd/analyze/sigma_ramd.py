"""
Run a sigma-RAMD calculation on the time profile data generated from
a RAMD log file, or perhaps some other means.
"""

import numpy as np
import scipy.integrate as sp_int

class functional_expansion_1d_ramd():
    def __init__(self, xi_bin_time_profiles, force_constant, beta, 
                 min_cv_val, max_cv_val, starting_cv_val):
        self.xi_bin_time_profiles = xi_bin_time_profiles
        [num_xi_bins, num_milestones, dummy] = xi_bin_time_profiles.shape
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
        self.h_xi = self.xi_span / (self.n_xi-1)
        self.z_s = np.linspace(self.a, self.c, self.n)
        self.xi_s = np.linspace(self.xi_a, self.xi_b, self.n_xi)
        return
    
    
    def make_zeroth_order_terms(self):
        # inverse times for each slice of xi
        self.flux_array = np.zeros(self.n_xi)
        start_index = int((self.b - self.a)/self.h_z)
        #print("start_index:", start_index)
        for i, xi in enumerate(self.xi_s):
            time_start_to_finish = self.xi_bin_time_profiles[i, -1, start_index]
            #print("i:", i, "time_start_to_finish:", time_start_to_finish)
            self.flux_array[i] = 1.0 / time_start_to_finish
            
        self.T = self.xi_span / sp_int.simps(self.flux_array, dx=self.h_xi)
        #print("T:", self.T)
        
    def make_first_order_terms(self):
        time_integrals_xi = np.zeros(self.n_xi)
        for i, xi in enumerate(self.xi_s):
            time_span_b_to_z = np.zeros(self.n)
            for j, z in enumerate(self.z_s):
                time_span_b_to_z[j] = self.xi_bin_time_profiles[i, j, 0]
            integrated_time_b_to_z = sp_int.simps(time_span_b_to_z, dx=self.h_z)
            
            boundary_term1 = (self.c-self.a)*self.xi_bin_time_profiles[i, -1, 0]
            
            time_span_z_to_c = np.zeros(self.n)
            for j, z in enumerate(self.z_s):
                time_span_z_to_c[j] = self.xi_bin_time_profiles[i, -1, j]
            integrated_time_z_to_c = sp_int.simps(time_span_z_to_c, dx=self.h_z)
            
            time_integrals_xi[i] \
                = (boundary_term1 - integrated_time_b_to_z \
                   - integrated_time_z_to_c) * self.flux_array[i]**2 * xi
            
        time_integral = sp_int.simps(time_integrals_xi, dx=self.h_xi) \
            * self.T**2 / self.xi_span
        self.first_order_correction = -self.beta*self.A*time_integral*self.T
        
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
            
            time_span_off_diag_b_to_z = np.zeros(self.n)
            for j1, z in enumerate(self.z_s):
                integral_span_inner1 = np.zeros(j1+1)
                for j2 in range(j1+1):
                   
                    integral_span_inner1[j2] \
                        = self.xi_bin_time_profiles[i, j2, 0]
                
                time_b_to_z = sp_int.simps(integral_span_inner1, dx=self.h_z)
                time_span_off_diag_b_to_z[j1] = time_b_to_z
            
            integrated_time_b_to_z \
                = sp_int.simps(time_span_off_diag_b_to_z, dx=self.h_z)
            
            # off diagonal boundary term c**2
            boundary_term_b_to_c1 = (self.c-self.a)**2 \
                * self.xi_bin_time_profiles[i, -1, 0]
            
            # off diagonal boundary term 2*c
            integral_span_inner1 = np.zeros(self.n)
            for j, z in enumerate(self.z_s):
                integral_span_inner1[j] = self.xi_bin_time_profiles[i, j, 0]
            boundary_term_b_to_c2 = 2.0 * (self.c-self.a) \
                * sp_int.simps(integral_span_inner1, dx=self.h_z)
            
            off_diag_b_to_z = boundary_term_b_to_c1 - boundary_term_b_to_c2 \
                + time_b_to_z
            
            # off diagonal z to c
            time_span_z_to_c = np.zeros(self.n)
            for j1, z in enumerate(self.z_s):
                integral_span_inner1 = np.zeros(self.n-j1)
                for j2 in range(j1, self.n):
                    index2 = self.n-j2-1
                    
                    integral_span_inner1[index2] \
                        = self.xi_bin_time_profiles[i, -1, j2]
                
                time_z_to_c = sp_int.simps(integral_span_inner1, dx=self.h_z)
                time_span_z_to_c[j1] = time_z_to_c
            
            time_z_to_c = sp_int.simps(time_span_z_to_c, dx=self.h_z)
            off_diag_est = off_diag_b_to_z + time_z_to_c
            time_integrals_xi[i] = (-on_diag_est + off_diag_est) \
                * self.flux_array[i]**2 * xi
        
        # Down here, integrate over u values
        time_integral = sp_int.simps(time_integrals_xi, dx=self.h_xi) \
            * self.T**2 / self.xi_span
        #print("time_integral o2:", time_integral)
        # TODO: missing a lot of stuff here...
        self.second_order_correction = self.beta**2*self.A**2*time_integral/2.0
        
    def make_second_order_time_estimate(self):
        self.make_zeroth_order_terms()
        self.make_first_order_terms()
        self.make_second_order_terms()
        time_estimate = self.T*np.exp(self.first_order_correction)\
            *np.exp(self.second_order_correction)
        return time_estimate