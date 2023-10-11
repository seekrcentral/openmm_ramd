"""
Base functions, classes, and constants for openmm_ramd analysis.
"""

import PyGT
import numpy as np
from openmm import unit
import matplotlib.pyplot as plt

import openmm_ramd.analyze.milestoning as milestoning
import openmm_ramd.analyze.remove_slant as remove_slant
import openmm_ramd.analyze.error_margins as error_margins

kB_kcal_per_mol_per_kelvin = unit.MOLAR_GAS_CONSTANT_R.in_units_of(
    unit.kilocalories_per_mole / unit.kelvin)
A_per_nm = 10.0

class Data_sample():
    """
    
    """
    def __init__(self):
        self.N_ij = None
        self.R_i = None
        self.t = None
        self.Q = None
        self.K = None
        self.K_hat = None
        self.p_i = None
        self.free_energy_profile = None
        self.diffusion_profile = None
        self.MFPT = None
        self.k_off = None

class Xi_bin():
    """
    
    """
    def __init__(self, num_milestones, milestones, lowest_boundary, midpoint, 
                 highest_boundary, mode=1):
        self.num_milestones = num_milestones
        self.milestones = milestones
        self.lowest_boundary = lowest_boundary
        self.midpoint = midpoint
        self.highest_boundary = highest_boundary
        self.trajs = []
        self.main_data_sample = None
        self.main_data_sample_unslanted = None
        self.data_samples = []
        self.data_samples_unslanted = []
        self.counts = None
        self.num_error_samples = 0
        self.mode = mode
        return
    
    def assign_main_data_sample(self, count_matrix, time_vector, 
                                avg_time_vector):
        self.main_data_sample = Data_sample()
        self.main_data_sample.N_ij = count_matrix
        self.main_data_sample.R_i = time_vector
        self.main_data_sample.t = avg_time_vector
        self.main_data_sample.K = milestoning.make_transition_matrix(
            self.num_milestones, count_matrix)
        self.main_data_sample.Q = milestoning.make_rate_matrix(
            self.num_milestones, self.main_data_sample.K, avg_time_vector)
        return
    
    def sample_error_matrices(self, num_error_samples, stride_error_samples, 
                              skip_error_samples):
        self.num_error_samples = num_error_samples
        self.data_samples = error_margins.monte_carlo_milestoning_error(
            self.main_data_sample, num_error_samples, skip_error_samples, 
            stride_error_samples)
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
    
    def unslant_data_samples(self, forceRAMD_in_nm_unitless, beta):
        slanting_force = -self.midpoint * forceRAMD_in_nm_unitless
        if self.mode == 1:
            unslanted_rate_matrix, unslanted_transition_matrix, \
                unslanted_avg_time_vector \
                    = remove_slant.unslant_transition_matrix_constant_approx(
                    self.milestones, self.main_data_sample.K, 
                    self.main_data_sample.t, slanting_force, beta)
            unslanted_energy_profile = np.zeros(self.num_milestones)
                    
        elif self.mode == 2:
            #print("data_sample.Q")
            #print(self.main_data_sample.Q)
            #print("data_sample.K")
            #print(self.main_data_sample.K)
            #print("self.main_data_sample.t")
            #print(self.main_data_sample.t)
                    
            unslanted_rate_matrix, unslanted_transition_matrix, \
                unslanted_avg_time_vector, unslanted_energy_profile \
                    = remove_slant.unslant_transition_matrix_from_energy_profile(
                    self.milestones, self.main_data_sample.free_energy_profile, 
                    self.main_data_sample.diffusion_profile, slanting_force, 
                    beta)
        
        else:
            raise Exception(f"Mode option not allowed: {self.mode}")
        
        new_data_sample = self.assign_unslanted_data(
            unslanted_avg_time_vector, unslanted_transition_matrix, 
            unslanted_rate_matrix)
        new_data_sample.free_energy_profile = unslanted_energy_profile
        self.main_data_sample_unslanted = new_data_sample
        
        if self.num_error_samples > 0:
            for i, data_sample in enumerate(self.data_samples):
                if self.mode == 1:
                    unslanted_rate_matrix, unslanted_transition_matrix, \
                        unslanted_avg_time_vector \
                        = remove_slant.unslant_transition_matrix_constant_approx(
                            self.milestones, data_sample.K, data_sample.t, 
                            slanting_force, beta)
                    unslanted_energy_profile = np.zeros(self.num_milestones)
                        
                elif self.mode == 2:
                    unslanted_rate_matrix, unslanted_transition_matrix, \
                        unslanted_avg_time_vector, unslanted_energy_profile \
                            = remove_slant.unslant_transition_matrix_from_energy_profile(
                            self.milestones, data_sample.free_energy_profile, 
                            data_sample.diffusion_profile, slanting_force, beta)
                    
                new_data_sample = self.assign_unslanted_data(
                    unslanted_avg_time_vector, unslanted_transition_matrix, 
                    unslanted_rate_matrix)
                new_data_sample.free_energy_profile = unslanted_energy_profile
                self.data_samples_unslanted.append(new_data_sample)
        
        return
    
    def make_profiles(self, starting_index, wrap_sink, unitless_beta,
                              data_sample=None):
        h = self.milestones[1] - self.milestones[0]
        if data_sample is None:
            data_sample = self.main_data_sample
        data_sample.p_i = milestoning.compute_stationary_probabilities(
            data_sample.K, data_sample.t, 
            self.num_milestones, starting_index, wrap_sink=wrap_sink)
        data_sample.free_energy_profile = remove_slant.make_energy_profile(
            data_sample.p_i, unitless_beta, starting_index)
        data_sample.diffusion_profile = milestoning.compute_diffusion_coefficients(
            data_sample.Q, self.num_milestones, h)
    
class Analysis:
    """
    
    """
    def __init__(self, num_milestones, milestones, h):
        self.num_milestones = num_milestones
        self.milestones = milestones
        self.main_data_sample = None
        self.data_samples = []
        self.p_i = None
        self.p_i_error = None
        self.free_energy_profile = None
        self.free_energy_profile_err = None
        self.diffusion_profile = None
        self.diffusion_profile_err = None
        self.MFPT = None
        self.MFPT_error = None
        self.k_off = None
        self.k_off_error = None
        self.h = h
        return
    
    def avg_unslanted_xi_bins(self, xi_bins):
        # First the main data sample
        rate_matrix_sum = np.zeros((self.num_milestones, self.num_milestones))
        transition_matrix_sum = np.zeros((self.num_milestones, self.num_milestones))
        avg_time_vector_sum = np.zeros((self.num_milestones, 1))
        counts_sum = np.zeros(self.num_milestones)
        for i, xi_bin in enumerate(xi_bins):
            for j in range(self.num_milestones):
                rate_matrix_sum[:, j] += xi_bin.counts[j] \
                    * xi_bin.main_data_sample_unslanted.Q[:, j]
                transition_matrix_sum[:, j] += xi_bin.counts[j] \
                    * xi_bin.main_data_sample_unslanted.K[:, j]
                avg_time_vector_sum[j, 0] += xi_bin.counts[j] \
                    * xi_bin.main_data_sample_unslanted.t[j, 0]
                counts_sum[j] += xi_bin.counts[j]
            
        rate_matrix_avg = np.zeros((self.num_milestones, self.num_milestones))
        transition_matrix_avg = np.zeros((self.num_milestones, self.num_milestones))
        avg_time_vector_avg = np.zeros((self.num_milestones, 1))
        for j in range(self.num_milestones):
            if counts_sum[j] == 0:
                rate_matrix_avg[j-1, j] = rate_matrix_avg[j, j-1]
                rate_matrix_avg[j, j] = -rate_matrix_avg[j, j-1]
            else:
                rate_matrix_avg[:, j] = rate_matrix_sum[:, j] / counts_sum[j]
                transition_matrix_avg[:, j] = transition_matrix_sum[:, j] / counts_sum[j]
                avg_time_vector_avg[j, 0] = avg_time_vector_sum[j, 0] / counts_sum[j]
        
        correct_milestoning_model_for_ramd(
            rate_matrix_avg, transition_matrix_avg, avg_time_vector_avg)
        
        self.main_data_sample = Data_sample()
        self.main_data_sample.t = avg_time_vector_avg
        self.main_data_sample.K = transition_matrix_avg
        self.main_data_sample.Q = rate_matrix_avg
        
        # Next the other data samples
        self.num_error_samples = len(xi_bins[0].data_samples)
        
        if self.num_error_samples > 0:
            for k in range(self.num_error_samples):
                rate_matrix_sum = np.zeros((self.num_milestones, self.num_milestones))
                transition_matrix_sum = np.zeros((self.num_milestones, self.num_milestones))
                avg_time_vector_sum = np.zeros((self.num_milestones, 1))
                counts_sum = np.zeros(self.num_milestones)
                for i, xi_bin in enumerate(xi_bins):
                    for j in range(self.num_milestones):
                        rate_matrix_sum[:, j] += xi_bin.counts[j] \
                            * xi_bin.data_samples_unslanted[k].Q[:, j]
                        transition_matrix_sum[:, j] += xi_bin.counts[j] \
                            * xi_bin.data_samples_unslanted[k].K[:, j]
                        avg_time_vector_sum[j, 0] += xi_bin.counts[j] \
                            * xi_bin.data_samples_unslanted[k].t[j, 0]
                        counts_sum[j] += xi_bin.counts[j]
                    
                rate_matrix_avg = np.zeros((self.num_milestones, self.num_milestones))
                transition_matrix_avg = np.zeros((self.num_milestones, self.num_milestones))
                avg_time_vector_avg = np.zeros((self.num_milestones, 1))
                for j in range(self.num_milestones):
                    if counts_sum[j] == 0:
                        rate_matrix_avg[j-1, j] = rate_matrix_avg[j, j-1]
                        rate_matrix_avg[j, j] = -rate_matrix_avg[j, j-1]
                    else:
                        rate_matrix_avg[:, j] = rate_matrix_sum[:, j] / counts_sum[j]
                        transition_matrix_avg[:, j] = transition_matrix_sum[:, j] / counts_sum[j]
                        avg_time_vector_avg[j, 0] = avg_time_vector_sum[j, 0] / counts_sum[j]
                
                correct_milestoning_model_for_ramd(
                    rate_matrix_avg, transition_matrix_avg, avg_time_vector_avg)
                
                new_data_sample = Data_sample()
                new_data_sample.t = avg_time_vector_avg
                new_data_sample.K = transition_matrix_avg
                new_data_sample.Q = rate_matrix_avg
                self.data_samples.append(new_data_sample)
                
        return
    
    def make_profiles(self, starting_index, wrap_sink, unitless_beta):
        self.p_i = milestoning.compute_stationary_probabilities(
            self.main_data_sample.K, self.main_data_sample.t, 
            self.num_milestones, starting_index, wrap_sink=wrap_sink)
        self.free_energy_profile = remove_slant.make_energy_profile(
            self.p_i, unitless_beta, starting_index)
        self.diffusion_profile = milestoning.compute_diffusion_coefficients(
            self.main_data_sample.Q, self.num_milestones, self.h)
        p_i_list = []
        free_energy_profile_list = []
        diffusion_profile_list = []
        if self.num_error_samples > 0:
            for k, data_sample in enumerate(self.data_samples):
                p_i = milestoning.compute_stationary_probabilities(
                    data_sample.K, data_sample.t, self.num_milestones, 
                    starting_index, wrap_sink=wrap_sink)
                free_energy_profile = remove_slant.make_energy_profile(
                    p_i, unitless_beta, starting_index)
                diffusion_profile = milestoning.compute_diffusion_coefficients(
                    data_sample.Q, self.num_milestones, self.h)
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
        
    def make_times(self, starting_index, bulk_milestone):
        # Recompute main diagonal
        for j in range(self.num_milestones):
            self.main_data_sample.Q[j, j] = 0.0
            self.main_data_sample.Q[j, j] = -sum(self.main_data_sample.Q[:,j])
        
        
        time_estimate = compute_time(self.main_data_sample.Q, starting_index, 
                                     bulk_milestone)
        if self.num_error_samples > 0:
            time_estimate_list = []
            k_off_list = []
            for k, data_sample in enumerate(self.data_samples):
                tmp_float128 = data_sample.Q.astype(np.longdouble)
                for j in range(self.num_milestones):
                    data_sample.Q[j, j] = 0.0
                    data_sample.Q[j, j] = -sum(data_sample.Q[:,j])
                
                try:
                    time_estimate_val = compute_time(
                        tmp_float128, starting_index, bulk_milestone)
                    time_estimate_list.append(float(time_estimate_val))
                    k_off_list.append(1.0 / time_estimate_val)
                except AssertionError:
                    print(f"Unstable sampled error matrix in iteration {k}. Skipping")
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
    
    def create_plots(self):
        pi_fig, ax1 = plt.subplots()
        ax1.errorbar(np.round(self.milestones[:-1], 3), self.free_energy_profile[:-1], 
                     yerr=self.free_energy_profile_err[:-1], color="r", ecolor="r", 
                     capsize=2)
        ax1.set_xticks(np.round(self.milestones[:-1], 3), 
                       np.round(self.milestones[:-1], 3), rotation=75)
        ax1.set_ylabel("\u0394G(z) (kcal/mol)", color="r")
        ax1.set_xlabel("z(nm)")
        
        #ax2 = ax1.twinx()
        #ax2.set_ylabel("D(z) ($nm^2/s$)", color="b")
        #ax2.errorbar(np.round(self.milestones[:-2], 3), self.diffusion_profile[:-2], 
        #             yerr=self.diffusion_profile_err[:-2], color="b", ecolor="b", 
        #             capsize=2)
        #ax2.set_ylim(0.0, 1.2*np.max(self.diffusion_profile))
        plt.tight_layout()
        #pi_fig.savefig(os.path.join(image_directory, "profile.png"))
        plt.show()
        
def make_xi_bins(num_xi_bins, num_milestones, milestones, mode):
    xi_bins = []
    xi_bins_ranges = np.linspace(-1.0, 1.0, num_xi_bins+1)
    for j in range(num_xi_bins):
        midpoint = 0.5 * (xi_bins_ranges[j] + xi_bins_ranges[j+1])
        #xi_values.append(xi_value)
        xi_bin = Xi_bin(num_milestones, milestones, xi_bins_ranges[j], midpoint, 
                        xi_bins_ranges[j+1], mode)
        xi_bins.append(xi_bin)
        
    return xi_bins

def compute_time(rate_matrix, starting_index, bulk_milestone):
    #rate_matrix = remove_slant.milestoning_statistics_from_profiles(
    #    milestones, energy_profile_avg, diffusion_profile_avg, beta)
    B, tau = PyGT.tools.load_CTMC(rate_matrix)
    mfpt_ij, mfpt_ji = PyGT.mfpt.compute_MFPT(
                    starting_index, bulk_milestone, B, tau)
    time_estimate = mfpt_ji
        
    return time_estimate

def correct_rate_matrix_for_ramd(rate_matrix_avg):
    end_rate = rate_matrix_avg[-3, -2]
    """
    rate_matrix_avg[-1, -2] = 0.5 * end_rate
    rate_matrix_avg[-3, -2] = 0.5 * end_rate
    rate_matrix_avg[-2, -2] = -end_rate
    rate_matrix_avg[-2, -1] = 0.5 * end_rate
    rate_matrix_avg[-1, -1] = -0.5 * end_rate
    """
    return

def correct_transition_matrix_for_ramd(transition_matrix_avg):
    """
    transition_matrix_avg[-1, -2] = 0.5
    transition_matrix_avg[-3, -2] = 0.5
    transition_matrix_avg[-2, -1] = 1.0
    """
    return

def correct_avg_time_vector_for_ramd(avg_time_vector_avg):
    avg_time_vector_avg[-1] = avg_time_vector_avg[-2]
    return

def correct_milestoning_model_for_ramd(rate_matrix_avg, transition_matrix_avg,
                                       avg_time_vector_avg):
    correct_rate_matrix_for_ramd(rate_matrix_avg)
    correct_transition_matrix_for_ramd(transition_matrix_avg)
    correct_avg_time_vector_for_ramd(avg_time_vector_avg)
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
