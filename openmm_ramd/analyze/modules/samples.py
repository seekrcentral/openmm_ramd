"""
Base functions, classes, and constants for openmm_ramd analysis.
"""

import os
from copy import deepcopy

import PyGT
import numpy as np
from openmm import unit
import matplotlib.pyplot as plt

import openmm_ramd.analyze.modules.milestoning as milestoning
import openmm_ramd.analyze.modules.remove_slant as remove_slant
import openmm_ramd.analyze.modules.error_margins as error_margins

kB_kcal_per_mol_per_kelvin = unit.MOLAR_GAS_CONSTANT_R.in_units_of(
    unit.kilocalories_per_mole / unit.kelvin)

MAX_PLOT_ENERGY = 35.0 # kcal/mol
PLOT_MARGIN = 4.0 # kcal/mol
FAST_RATE = 1e9
TINY_RATE = 1e-9
TINY_PROB = 1e-9

class Data_sample():
    """
    
    """
    def __init__(self):
        self.N_ij = None
        self.R_i = None
        self.t = None
        self.t_matrix = None
        self.Q = None
        self.K = None
        self.K_hat = None
        self.p_i = None
        self.free_energy_profile = None
        self.diffusion_profile = None
        self.MFPT = None
        self.k_off = None

class Xi_bin_traj():
    """
    
    """
    def __init__(self, index, num_milestones, milestones, lowest_boundary,
                 midpoint, highest_boundary, situation):
        self.trajs = []
        self.index = index
        self.num_milestones = num_milestones
        self.milestones = milestones
        self.lowest_boundary = lowest_boundary
        self.midpoint = midpoint
        self.highest_boundary = highest_boundary
        self.situation = situation

class Xi_bin():
    """
    
    """
    def __init__(self, index, num_milestones, milestones, lowest_boundary, 
                 midpoint, highest_boundary, situation, debug_directory=None):
        self.index = index
        self.num_milestones = num_milestones
        self.milestones = milestones
        self.lowest_boundary = lowest_boundary
        self.midpoint = midpoint
        self.highest_boundary = highest_boundary
        self.count_matrix = np.zeros((num_milestones, num_milestones))
        self.time_vector = np.zeros((num_milestones, 1))
        self.time_matrix = np.zeros((num_milestones, num_milestones))
        self.main_data_sample = None
        self.main_data_sample_unslanted = None
        self.counts = None
        self.situation = situation
        self.debug_directory = debug_directory
        return
    
    def assign_main_data_sample(self, count_matrix, time_vector, 
                                avg_time_vector, avg_time_matrix):
        self.main_data_sample = Data_sample()
        self.main_data_sample.N_ij = count_matrix
        self.main_data_sample.R_i = time_vector
        self.main_data_sample.t = avg_time_vector
        self.main_data_sample.t_matrix = avg_time_matrix
        self.main_data_sample.K = milestoning.make_transition_matrix(
            self.num_milestones, count_matrix)
        self.main_data_sample.Q = milestoning.make_rate_matrix(
            self.num_milestones, self.main_data_sample.K, avg_time_vector)
        if self.debug_directory is not None:
            K_matrix_filename = os.path.join(
                self.debug_directory, f"slanted_K_matrix_{self.index}.txt")
            np.savetxt(K_matrix_filename, self.main_data_sample.K, fmt="%.3e")
            Q_matrix_filename = os.path.join(
                self.debug_directory, f"slanted_Q_matrix_{self.index}.txt")
            np.savetxt(Q_matrix_filename, self.main_data_sample.Q, fmt="%.3e")
        
        return
    
    def make_profiles(self, starting_index, wrap_sink, unitless_beta,
                              data_sample=None):
        h = self.milestones[1] - self.milestones[0]
        if data_sample is None:
            using_main = True
            data_sample = self.main_data_sample
        else:
            using_main = False
        milestoning.fill_gaps_in_matrix(data_sample.K, normalize=True)
        data_sample.p_i = milestoning.compute_stationary_probabilities(
            data_sample.K, data_sample.t, self.num_milestones, 
            starting_index, self.situation, wrap_sink=wrap_sink)
        data_sample.free_energy_profile = remove_slant.make_energy_profile(
            data_sample.p_i, unitless_beta, starting_index)
        
        data_sample.diffusion_profile = milestoning.compute_diffusion_coefficients(
            data_sample.Q, data_sample.K, data_sample.free_energy_profile, 
            self.milestones, unitless_beta)
        
        if using_main and self.debug_directory is not None:
            p_i_vector_filename = os.path.join(
                self.debug_directory, f"p_i_{self.index}.txt")
            np.savetxt(p_i_vector_filename, data_sample.p_i, fmt="%.3e")
            free_energy_vector_filename = os.path.join(
                self.debug_directory, f"slanted_free_energy_{self.index}.txt")
            np.savetxt(free_energy_vector_filename, 
                       data_sample.free_energy_profile, fmt="%.3e")
            diffusion_vector_filename = os.path.join(
                self.debug_directory, f"diffusion_{self.index}.txt")
            np.savetxt(diffusion_vector_filename, data_sample.diffusion_profile,
                       fmt="%.3e")
        
        return
    
    def assign_unslanted_data(
            self, unslanted_avg_time_vector, 
            unslanted_transition_matrix, unslanted_rate_matrix):
        data_sample = Data_sample()
        data_sample.N_ij = self.main_data_sample.N_ij
        data_sample.R_i = self.main_data_sample.R_i
        data_sample.t = unslanted_avg_time_vector
        data_sample.K = unslanted_transition_matrix
        data_sample.Q = unslanted_rate_matrix
        return data_sample
    
    def unslant_data_samples(self, forceRAMD_in_A_unitless, beta):
        slanting_force = -self.midpoint * forceRAMD_in_A_unitless
        unslanted_rate_matrix, unslanted_transition_matrix, \
            unslanted_avg_time_vector \
                = remove_slant.unslant_transition_matrix_constant_approx(
                self.milestones, self.main_data_sample.K, 
                self.main_data_sample.t, self.main_data_sample.t_matrix, 
                slanting_force, beta)
        unslanted_energy_profile = np.zeros(self.num_milestones)
        new_data_sample = self.assign_unslanted_data(
            unslanted_avg_time_vector, unslanted_transition_matrix, 
            unslanted_rate_matrix)
        new_data_sample.N_ij = self.main_data_sample.N_ij
        new_data_sample.R_i = self.main_data_sample.R_i
        new_data_sample.free_energy_profile = unslanted_energy_profile
        self.main_data_sample_unslanted = new_data_sample
        if self.debug_directory is not None:
            K_matrix_filename = os.path.join(
                self.debug_directory, f"unslanted_K_matrix_{self.index}.txt")
            np.savetxt(K_matrix_filename, unslanted_transition_matrix, fmt="%.3e")
            Q_matrix_filename = os.path.join(
                self.debug_directory, f"unslanted_Q_matrix_{self.index}.txt")
            np.savetxt(Q_matrix_filename, unslanted_rate_matrix, fmt="%.3e")
            avg_time_vector_filename = os.path.join(
                self.debug_directory, 
                f"unslanted_avg_time_vector_{self.index}.txt")
            np.savetxt(avg_time_vector_filename, unslanted_avg_time_vector,
                       fmt="%.3e")
        return
    
class Analysis:
    """
    
    """
    def __init__(self, num_milestones, milestones, h, situation, 
                 bulk_milestones, verbose=False, debug_directory=None):
        self.num_milestones = num_milestones
        self.milestones = milestones
        self.main_data_sample = None
        self.data_samples = []
        self.counts_sum = None
        self.p_i = None
        self.p_i_error = None
        self.free_energy_profile = None
        self.free_energy_profile_err = None
        self.diffusion_profile = None
        self.diffusion_profile_err = None
        self.bulk_milestones = bulk_milestones
        self.MFPT = None
        self.MFPT_error = None
        self.k_off = None
        self.k_off_error = None
        self.h = h
        self.num_error_samples = 0
        self.situation = situation
        self.verbose = verbose
        self.debug_directory = debug_directory
        return
    
    def avg_unslanted_xi_bins(self, xi_bins):
        # First the main data sample
        rate_matrix_sum = np.zeros((self.num_milestones, self.num_milestones))
        transition_matrix_sum = np.zeros((self.num_milestones, self.num_milestones))
        avg_time_vector_sum = np.zeros((self.num_milestones, 1))
        count_matrix_sum = np.zeros((self.num_milestones, self.num_milestones))
        total_time_vector_sum = np.zeros((self.num_milestones, 1))
        counts_sum = np.zeros(self.num_milestones)
        for i, xi_bin in enumerate(xi_bins):
            for j in range(self.num_milestones):
                rate_matrix_sum[:, j] += xi_bin.counts[j] \
                    * xi_bin.main_data_sample_unslanted.Q[:, j]
                transition_matrix_sum[:, j] += xi_bin.counts[j] \
                    * xi_bin.main_data_sample_unslanted.K[:, j]
                avg_time_vector_sum[j, 0] += xi_bin.counts[j] \
                    * xi_bin.main_data_sample_unslanted.t[j, 0]
                count_matrix_sum[:, j] += \
                    xi_bin.main_data_sample_unslanted.N_ij[:, j]
                total_time_vector_sum[j, 0] += \
                    xi_bin.main_data_sample_unslanted.R_i[j, 0]
                counts_sum[j] += xi_bin.counts[j]
        
        rate_matrix_avg = np.zeros((self.num_milestones, self.num_milestones))
        transition_matrix_avg = np.zeros((self.num_milestones, self.num_milestones))
        avg_time_vector_avg = np.zeros((self.num_milestones, 1))
        for j in range(self.num_milestones):
            #if counts_sum[j] == 0:
            #    rate_matrix_avg[j-1, j] = rate_matrix_avg[j, j-1]
            #    rate_matrix_avg[j, j] = -rate_matrix_avg[j, j-1]
            #else:
            if counts_sum[j] > 0:
                rate_matrix_avg[:, j] = rate_matrix_sum[:, j] / counts_sum[j]
                transition_matrix_avg[:, j] = transition_matrix_sum[:, j] / counts_sum[j]
                avg_time_vector_avg[j, 0] = avg_time_vector_sum[j, 0] / counts_sum[j]
        
        if self.debug_directory:
            Q_gaps_filename = os.path.join(self.debug_directory, 
                                           "Q_all_gaps.txt")
            np.savetxt(Q_gaps_filename, rate_matrix_avg, fmt="%.3e")
            K_gaps_filename = os.path.join(self.debug_directory, 
                                           "K_all_gaps.txt")
            np.savetxt(K_gaps_filename, transition_matrix_avg, fmt="%.3e")
            time_gaps_filename = os.path.join(self.debug_directory, 
                                           "avg_time_vector_all_gaps.txt")
            np.savetxt(time_gaps_filename, avg_time_vector_avg, fmt="%.3e")
            counts_sum_filename = os.path.join(self.debug_directory,
                                               "counts_all.txt")
            np.savetxt(counts_sum_filename, counts_sum, fmt="%d")
        
        self.counts_sum = counts_sum
        milestoning.fill_gaps_in_matrix(rate_matrix_avg, normalize=False, flat=True)
        milestoning.fill_gaps_in_matrix(transition_matrix_avg, normalize=True, flat=True)
        milestoning.fill_gaps_in_vector(avg_time_vector_avg)
        
        if self.debug_directory:
            Q_all_filename = os.path.join(self.debug_directory, 
                                           "Q_all.txt")
            np.savetxt(Q_all_filename, rate_matrix_avg, fmt="%.3e")
            K_all_filename = os.path.join(self.debug_directory, 
                                           "K_all.txt")
            np.savetxt(K_all_filename, transition_matrix_avg, fmt="%.3e")
            time_all_filename = os.path.join(self.debug_directory, 
                                           "avg_time_vector_all.txt")
            np.savetxt(time_all_filename, avg_time_vector_avg, fmt="%.3e")
            
        
        # TODO: remove
        #correct_milestoning_model_for_ramd(
        #    rate_matrix_avg, transition_matrix_avg, avg_time_vector_avg)
        
        self.main_data_sample = Data_sample()
        self.main_data_sample.t = avg_time_vector_avg
        self.main_data_sample.K = transition_matrix_avg
        self.main_data_sample.Q = rate_matrix_avg
        self.main_data_sample.N_ij = count_matrix_sum
        self.main_data_sample.R_i = total_time_vector_sum
        
        return
    
    def estimate_matrix_beyond_zero_count(self, starting_index):
        found_zero_count = False
        rate_estimate = FAST_RATE
        if (self.situation == 1) or (self.situation == 3):
            for i in range(starting_index, self.num_milestones):
                if self.counts_sum[i] < 1:
                    found_zero_count = True
                    rate_estimate = np.sum(self.main_data_sample.Q[:,i])
                    
                if found_zero_count:
                    if i == 0:
                        self.main_data_sample.K[i+1,i] = 1.0
                        self.main_data_sample.Q[i+1,i] = rate_estimate
                        #self.main_data_sample.Q[i,i] = -rate_estimate
                    
                    elif i == self.num_milestones-1:
                        self.main_data_sample.K[i-1,i] = 1.0
                        self.main_data_sample.Q[i-1,i] = rate_estimate
                        #self.main_data_sample.Q[i,i] = -rate_estimate
                        
                    else:
                        self.main_data_sample.K[i-1,i] = 0.45
                        self.main_data_sample.K[i+1,i] = 0.55
                        self.main_data_sample.Q[i-1,i] = 0.45 * rate_estimate
                        self.main_data_sample.Q[i+1,i] = 0.55 * rate_estimate
                        #self.main_data_sample.Q[i,i] = -rate_estimate
        
        found_zero_count = False
        rate_estimate = FAST_RATE
        if (self.situation == 2) or (self.situation == 3):
            for i in range(starting_index, -1, -1):
                if self.counts_sum[i] == 0:
                    found_zero_count = True
                    rate_estimate = np.sum(self.main_data_sample.Q[:,i])
                    
                if found_zero_count:
                    if i == 0:
                        self.main_data_sample.K[i+1,i] = 1.0
                        self.main_data_sample.Q[i+1,i] = rate_estimate
                        #self.main_data_sample.Q[i,i] = -rate_estimate
                    
                    elif i == self.num_milestones-1:
                        self.main_data_sample.K[i-1,i] = 1.0
                        self.main_data_sample.Q[i-1,i] = rate_estimate
                        #self.main_data_sample.Q[i,i] = -rate_estimate
                        
                    else:
                        self.main_data_sample.K[i-1,i] = 0.55
                        self.main_data_sample.K[i+1,i] = 0.45
                        self.main_data_sample.Q[i-1,i] = 0.55 * rate_estimate
                        self.main_data_sample.Q[i+1,i] = 0.45 * rate_estimate
                        #self.main_data_sample.Q[i,i] = -rate_estimate
                
        if self.situation not in [1, 2, 3]:
            raise Exception(f"Not yet implemented: situation {self.situation}")
        
        if self.debug_directory:
            Q_flat_filename = os.path.join(self.debug_directory, "Q_flat.txt")
            np.savetxt(Q_flat_filename, self.main_data_sample.Q, fmt="%.3e")
            K_flat_filename = os.path.join(self.debug_directory, "K_flat.txt")
            np.savetxt(K_flat_filename, self.main_data_sample.K, fmt="%.3e")
        
        return
    
    def sample_error_matrices(self, num_error_samples, stride_error_samples, 
                              skip_error_samples):
        self.num_error_samples = num_error_samples
        self.data_samples = error_margins.monte_carlo_milestoning_error(
            self.main_data_sample, num_error_samples, skip_error_samples, 
            stride_error_samples, self.verbose)
        return
    
    def make_profiles(self, starting_index, wrap_sink, unitless_beta):
        milestoning.fill_gaps_in_matrix(self.main_data_sample.K, normalize=True)
        self.p_i = milestoning.compute_stationary_probabilities(
            self.main_data_sample.K, self.main_data_sample.t, 
            self.num_milestones, starting_index, self.situation, 
            wrap_sink=wrap_sink)
        
        self.free_energy_profile = remove_slant.make_energy_profile(
            self.p_i, unitless_beta, starting_index)
        #self.diffusion_profile = milestoning.compute_diffusion_coefficients(
        #    self.main_data_sample.Q, self.num_milestones, self.h)
        self.diffusion_profile = milestoning.compute_diffusion_coefficients(
            self.main_data_sample.Q, self.main_data_sample.K, 
            self.free_energy_profile, self.milestones, unitless_beta)
        p_i_list = []
        free_energy_profile_list = []
        diffusion_profile_list = []
        if self.num_error_samples > 0:
            for k, data_sample in enumerate(self.data_samples):
                p_i = milestoning.compute_stationary_probabilities(
                    data_sample.K, data_sample.t, self.num_milestones, 
                    starting_index, self.situation, wrap_sink=wrap_sink)
                free_energy_profile = remove_slant.make_energy_profile(
                    p_i, unitless_beta, starting_index)
                #diffusion_profile = milestoning.compute_diffusion_coefficients(
                #    data_sample.Q, self.num_milestones, self.h)
                diffusion_profile = milestoning.compute_diffusion_coefficients(
                    data_sample.Q, data_sample.K, free_energy_profile, 
                    self.milestones, unitless_beta)
                p_i_list.append(p_i)
                free_energy_profile_list.append(free_energy_profile)
                diffusion_profile_list.append(diffusion_profile)
        
        self.p_i_error = np.zeros(self.p_i.shape)
        self.free_energy_profile_err = np.zeros(
            self.free_energy_profile.shape)
        self.diffusion_profile_err = np.zeros(
            self.diffusion_profile.shape)
        if self.num_error_samples > 0:
            for i in range(self.p_i_error.shape[0]):
                p_i_val_list = []
                for j in range(len(p_i_list)):
                    p_i_val_list.append(p_i_list[j][i])
                self.p_i_error[i] = np.std(p_i_val_list)
            for i in range(self.free_energy_profile_err.shape[0]):
                free_energy_profile_val_list = []
                for j in range(len(free_energy_profile_list)):
                    free_energy_profile_val_list.append(free_energy_profile_list[j][i])
                self.free_energy_profile_err[i] = np.std(free_energy_profile_val_list)
            for i in range(self.diffusion_profile_err.shape[0]):
                diffusion_profile_val_list = []
                for j in range(len(diffusion_profile_list)):
                    diffusion_profile_val_list.append(diffusion_profile_list[j][i])
                self.diffusion_profile_err[i] = np.std(diffusion_profile_val_list)
                
        return
        
    def make_times(self, starting_index):
        # Recompute main diagonal
        for j in range(self.num_milestones):
            self.main_data_sample.Q[j, j] = 0.0
            column_sum = sum(self.main_data_sample.Q[:,j])
            assert column_sum >= 0.0, \
                "Negative off-diagonal values in rate matrix not allowed"
            #if column_sum == 0.0:
            #    if j > 0:
            #        self.main_data_sample.Q[j-1, j] += FAST_RATE
            #        self.main_data_sample.Q[j, j-1] += TINY_RATE
            #    if j < self.num_milestones-1:
            #        self.main_data_sample.Q[j+1, j] += FAST_RATE
            #        self.main_data_sample.Q[j, j+1] += TINY_RATE
        
        for j in range(self.num_milestones):
            self.main_data_sample.Q[j, j] = 0.0
            column_sum = sum(self.main_data_sample.Q[:,j])
            self.main_data_sample.Q[j, j] = -column_sum
        
        time_estimate = compute_time(
            self.main_data_sample.Q, starting_index, self.bulk_milestones)
        if self.num_error_samples > 0:
            time_estimate_list = []
            
            k_off_list = []
            for k, data_sample in enumerate(self.data_samples):
                tmp_float128 = data_sample.Q.astype(np.longdouble)
                
                for j in range(self.num_milestones):
                    data_sample.Q[j, j] = 0.0
                    column_sum = sum(data_sample.Q[:,j])
                    assert column_sum >= 0.0, \
                        "Negative off-diagonal values in rate matrix not allowed"
                    if column_sum == 0.0:
                        if j > 0:
                            data_sample.Q[j-1, j] += FAST_RATE
                            data_sample.Q[j, j-1] += TINY_RATE
                        if j < self.num_milestones-1:
                            data_sample.Q[j+1, j] += FAST_RATE
                            data_sample.Q[j, j+1] += TINY_RATE
                
                for j in range(self.num_milestones):
                    data_sample.Q[j, j] = 0.0
                    column_sum = sum(data_sample.Q[:,j])
                    data_sample.Q[j, j] = -column_sum
                
                try:
                    time_estimate_val = compute_time(
                        tmp_float128, starting_index, self.bulk_milestones)
                    time_estimate_list.append(float(time_estimate_val))
                    
                    k_off_list.append(1.0 / time_estimate_val)
                except AssertionError:
                    print(f"Unstable sampled error matrix in iter {k}. Skip.")
                    continue
            
            time_estimate_err = np.std(time_estimate_list)
            k_off_err = np.std(k_off_list)
        else:
            time_estimate_err = np.inf
            k_off_err = np.inf
            
        self.MFPT = time_estimate
        self.MFPT_error = time_estimate_err
        self.k_off = 1.0 / time_estimate
        self.k_off_error = k_off_err
        return
    
    def print_results(self):
        print("Printing results from RAMD analysis.")
        print("MFPT (ps):", pretty_string_value_error(
            self.MFPT, self.MFPT_error))
        print("k_off (1/ps):", pretty_string_value_error(
            self.k_off, self.k_off_error))
        return
    
    def create_plots(self, directory=None, show_plots=True):
        pi_fig, ax1 = plt.subplots(figsize=(10, 8))
        ax1.errorbar(np.round(self.milestones[:-1], 3), self.free_energy_profile[:-1], 
                     yerr=self.free_energy_profile_err[:-1], color="r", ecolor="r", 
                     capsize=2)
        upper_limit = np.min([np.max(self.free_energy_profile[:-1])+PLOT_MARGIN, 
                              MAX_PLOT_ENERGY])
        lower_limit = np.min(self.free_energy_profile[:-1])
        ax1.set_ylim(bottom=lower_limit, top=upper_limit)
        ax1.set_xticks(np.round(self.milestones[:-1], 3), 
                       np.round(self.milestones[:-1], 3), rotation=75)
        ax1.set_ylabel("W(z) (kcal/mol)", color="r")
        ax1.set_xlabel("z($\AA$)")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("D(z) ($nm^2/ps$)", color="b")
        ax2.errorbar(np.round(self.milestones[:-1], 3), self.diffusion_profile[:-1], 
                     yerr=self.diffusion_profile_err[:-1], color="b", ecolor="b", 
                     capsize=2)
        ax2.set_ylim(0.0, 1.2*np.max(self.diffusion_profile[1:-2]))
        
        ax3 = ax1.twinx()
        ax3.set_ylabel("counts", color="g")
        ax3.errorbar(np.round(self.milestones, 3), self.counts_sum, 
                     yerr=np.zeros(self.num_milestones), color="g", ecolor="g", 
                     capsize=0)
        ax3.set_ylim(1.0, 1.2*np.max(self.counts_sum))
        ax3.set_yscale("log", nonpositive="mask")
        ax3.spines['right'].set_position(('outward', 65))
        
        plt.tight_layout()
        
        if show_plots:
            if directory is None:
                plt.show()
            else:
                plt.savefig(os.path.join(directory, "pmf_diffusion.png"))
                
        return ax1, ax2, ax3
        
def make_xi_bins(num_xi_bins, num_milestones, milestones, situation, 
                 debug_directory=None):
    xi_bins = []
    xi_bins_ranges = np.linspace(-1.0, 1.0, num_xi_bins+1)
    for j in range(num_xi_bins):
        midpoint = 0.5 * (xi_bins_ranges[j] + xi_bins_ranges[j+1])
        #xi_values.append(xi_value)
        xi_bin = Xi_bin(j, num_milestones, milestones, xi_bins_ranges[j], 
                        midpoint, xi_bins_ranges[j+1], situation, 
                        debug_directory)
        xi_bins.append(xi_bin)
        
    return xi_bins

def make_xi_bin_trajs(num_xi_bins, num_milestones, milestones, situation):
    xi_bins = []
    xi_bins_ranges = np.linspace(-1.0, 1.0, num_xi_bins+1)
    for j in range(num_xi_bins):
        midpoint = 0.5 * (xi_bins_ranges[j] + xi_bins_ranges[j+1])
        #xi_values.append(xi_value)
        xi_bin = Xi_bin_traj(j, num_milestones, milestones, xi_bins_ranges[j], 
                        midpoint, xi_bins_ranges[j+1], situation)
        xi_bins.append(xi_bin)
        
    return xi_bins

def combine_dest_states(Q, src_indices, dest_indices):
    """
    When multiple milestones represent a given destination state, then
    adjust the rate matrix to consider all states, with fluxes into them,
    as a single state (single row and column in the matrix).
    """
    newQ = deepcopy(Q)
    new_src_indices = deepcopy(src_indices)
    new_dest_indices = deepcopy(dest_indices)
    n = newQ.shape[0]
    # insert a new row and column at the very end
    newQ = np.insert(newQ, n, np.zeros(n), 0)
    newQ = np.insert(newQ, n, np.zeros(n+1), 1)
    for i in range(len(dest_indices)):
        dest_index = new_dest_indices[i]
        dest_row = np.delete(newQ[dest_index,:], dest_index, 0)
        dest_column = np.delete(newQ[:,dest_index], dest_index, 0)
        newQ = np.delete(newQ, dest_index, 0)
        newQ = np.delete(newQ, dest_index, 1)
        newQ[-1,:] += dest_row
        newQ[:,-1] += dest_column
        
        for j in range(len(new_src_indices)):
            old_src_index = new_src_indices[j]
            if old_src_index > dest_index:
                new_src_indices[j] = old_src_index - 1
            else:
                new_src_indices[j] = old_src_index
        
        for j in range(len(new_dest_indices)):
            old_dest_index = new_dest_indices[j]
            if old_dest_index > dest_index:
                new_dest_indices[j] = old_dest_index - 1
            else:
                new_dest_indices[j] = old_dest_index
        
    n = newQ.shape[0]
    for i in range(n):
        newQ[i,i] = 0.0
        newQ[i,i] = -np.sum(newQ[i,:])
    
    return newQ, new_src_indices, n-1

def compute_time(rate_matrix, starting_index, bulk_milestones):
    bulk_combined_Q, new_starting_index_list, bulk_index \
        = combine_dest_states(rate_matrix.T, [starting_index], bulk_milestones)
    B, tau = PyGT.tools.load_CTMC(bulk_combined_Q.T)
    new_starting_index = new_starting_index_list[0]
    mfpt_ij, mfpt_ji = PyGT.mfpt.compute_MFPT(
        new_starting_index, bulk_index, B, tau)
    time_estimate = mfpt_ji
    return time_estimate

def correct_rate_matrix_for_ramd(rate_matrix_avg):
    #end_rate = rate_matrix_avg[-3, -2]
    """
    rate_matrix_avg[-1, -2] = 0.5 * end_rate
    rate_matrix_avg[-3, -2] = 0.5 * end_rate
    rate_matrix_avg[-2, -2] = -end_rate
    rate_matrix_avg[-2, -1] = 0.5 * end_rate
    rate_matrix_avg[-1, -1] = -0.5 * end_rate
    """
    num_rows = rate_matrix_avg.shape[0]
    for i in range(num_rows):
        if (np.sum(rate_matrix_avg[:i,i]) \
                + np.sum(rate_matrix_avg[(i+1):,i])) == 0:
            # If the rate going left or right is zero, then escape this state quickly
            if i > 0:
                rate_matrix_avg[i-1,i] = FAST_RATE
                rate_matrix_avg[i,i] -= FAST_RATE
            if i < num_rows-1:
                rate_matrix_avg[i+1,i] = FAST_RATE
                rate_matrix_avg[i,i] -= FAST_RATE
        elif np.sum(rate_matrix_avg[:i,i]) == 0:
            # If the rate going left is zero, then make it a slow rate left
            if i > 0:
                rate_matrix_avg[i-1,i] = TINY_RATE
                rate_matrix_avg[i,i] -= TINY_RATE
                
        elif np.sum(rate_matrix_avg[(i+1):,i]) == 0:
            # If the rate going left is zero, then make it a slow rate left
            if i < num_rows-1:
                rate_matrix_avg[i+1,i] = TINY_RATE
                rate_matrix_avg[i,i] -= TINY_RATE
                
    return

def correct_transition_matrix_and_avg_time_vector_for_ramd(
        transition_matrix_avg, avg_time_vector_avg):
    """
    transition_matrix_avg[-1, -2] = 0.5
    transition_matrix_avg[-3, -2] = 0.5
    transition_matrix_avg[-2, -1] = 1.0
    """
    num_rows = transition_matrix_avg.shape[0]
    for i in range(num_rows):
        if np.sum(transition_matrix_avg[:,i]) == 0:
            # If the prob going left or right is zero, then there is an equal escape probability
            if i == 0:
                transition_matrix_avg[i+1,i] = 1.0
                
            elif i == num_rows-1:
                transition_matrix_avg[i-1,i] = 1.0
                
            else:
                transition_matrix_avg[i-1,i] = 0.5
                transition_matrix_avg[i+1,i] = 0.5
            
            avg_time_vector_avg[i] = 1.0 / FAST_RATE
                
        elif np.sum(transition_matrix_avg[:i,i]) == 0:
            # If the prob going left is zero, then make it a slow rate left
            if i > 0:
                transition_matrix_avg[i-1,i] = TINY_PROB
                transition_matrix_avg[i+1,i] -= TINY_PROB
                
        elif np.sum(transition_matrix_avg[(i+1):,i]) == 0:
            # If the prob going left is zero, then make it a slow rate left
            if i < num_rows-1:
                transition_matrix_avg[i-1,i] -= TINY_PROB
                transition_matrix_avg[i+1,i] = TINY_PROB
    return

#def correct_avg_time_vector_for_ramd(avg_time_vector_avg):
#    avg_time_vector_avg[-1] = avg_time_vector_avg[-2]
#    return

# TODO: remove
def correct_milestoning_model_for_ramd(rate_matrix_avg, transition_matrix_avg,
                                       avg_time_vector_avg):
    #print("rate_matrix_avg:")
    #print(rate_matrix_avg)
    
    correct_rate_matrix_for_ramd(rate_matrix_avg)
    
    #print("rate_matrix_avg fixed:")
    #print(rate_matrix_avg)
    
    #print("transition_matrix_avg:")
    #print(transition_matrix_avg)
    #print("avg_time_vector_avg")
    #print(avg_time_vector_avg)
    
    correct_transition_matrix_and_avg_time_vector_for_ramd(
        transition_matrix_avg, avg_time_vector_avg)
    
    #print("transition_matrix_avg fixed:")
    #print(transition_matrix_avg)
    #print("avg_time_vector_avg fixed")
    #print(avg_time_vector_avg)
    #correct_avg_time_vector_for_ramd(avg_time_vector_avg)
    return

def pretty_string_value_error(value, error, error_digits=2, use_unicode=True):
    """
    Copied from seekr2
    
    Returns a value/error combination of numbers in a scientifically 
    'pretty' format.
    
    Scientific quantities often come as a *value* (the actual 
    quantity) and the *error* (the uncertainty in the value).
        
    Given two floats, value and error, return the two in a 
    'pretty' formatted string: where the value and error are truncated
    at the correct precision.
    
    Parameters
    ----------
    value : float
        The quantity in question
        
    error : float
        The uncertainty of the quantity
        
    error_digits : int, default 2
        How many significant figures the error has. Scientific 
        convention holds that errors have 1 or (at most) 2 significant
        figures. The larger number of digits is chosen here by default.
        
    Returns
    -------
    new_string : str
        A new list of strings sorted numerically
    
    Examples
    --------
    
        >>> pretty_string_value_error(1.23456789e8, 4.5678e5, 
                                      error_digits=2)
        "1.2346 +/- 0.0046 * 10^+08"
    
        >>> pretty_string_value_error(5.6e-2, 2.0e-3, error_digits=1)
        "5.6 +/- 0.2 * 10^-02"
    """
    if value is None:
        return "None"
    if error is None or not np.isfinite(error):
        if use_unicode:
            new_string = "{:.6E} \u00B1 UNKNOWN ERROR MARGIN".format(value)
        else:
            new_string = "{:.6E} +/- UNKNOWN ERROR MARGIN".format(value)
    else:
        if not np.isfinite(value):
            return str(value)
        assert "e" in "{:e}".format(value), "Cannot convert into scientific "\
            "notation: {1}".format(value)
        value_mantissa_str, value_exponent_str = \
            "{:e}".format(value).strip().split('e')
        value_mantissa = float(value_mantissa_str)
        value_exponent = int(value_exponent_str)
        error_mantissa_str, error_exponent_str = \
            "{:e}".format(error).strip().split('e')
        error_mantissa = float(error_mantissa_str)
        error_exponent = int(error_exponent_str)
        padding = value_exponent - error_exponent + error_digits - 1
        if padding < 1: padding = 1
        exp_diff = error_exponent - value_exponent
        string_for_formatting = "{:.%df}" % padding
        new_value_mantissa = string_for_formatting.format(value_mantissa)
        new_error_mantissa = string_for_formatting.format(
            error_mantissa*10**exp_diff)
        if use_unicode:
            new_string = "(%s \u00B1 %s) * 10^%s" % (
                new_value_mantissa, new_error_mantissa, value_exponent_str)
        else:
            new_string = "(%s +/- %s) * 10^%s" % (
                new_value_mantissa, new_error_mantissa, value_exponent_str)
    return new_string
