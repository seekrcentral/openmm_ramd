"""
conftest.py

configurations for openmm_ramd tests
"""

import os
import pytest

import openmm_ramd.analyze.parser as parser

TEST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

test_small_log = os.path.join(TEST_DIRECTORY, "data/small_linear.txt")
test_large_log = os.path.join(TEST_DIRECTORY, "data/large_linear.txt")

@pytest.fixture(scope="session")
def small_linear_logfile_persistent():
    """
    Create logfile output that is persistent across the tests.
    """
    trajectories, forceOutFreq, forceRAMD, timeStep, num_simulations, \
        temperature = parser.parse_ramd_log_file(test_small_log)
    return trajectories, forceOutFreq, forceRAMD, timeStep, num_simulations, \
        temperature

@pytest.fixture(scope="session")
def large_linear_logfile_persistent():
    """
    Create logfile output that is persistent across the tests.
    """
    trajectories, forceOutFreq, forceRAMD, timeStep, num_simulations, \
        temperature = parser.parse_ramd_log_file(test_large_log)
    return trajectories, forceOutFreq, forceRAMD, timeStep, num_simulations, \
        temperature