"""
Test openmm_ramd analyze functions.
"""

import os

import numpy as np

import utils
import openmm_ramd.analyze.sigma_ramd as sigma_ramd

TEST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
test_large_log = os.path.join(TEST_DIRECTORY, "data/large_linear.txt")

def test_sigma_ramd():
    my_analysis = sigma_ramd.sigma_ramd([test_large_log], num_angle_bins=2)
    my_analysis.print_results()