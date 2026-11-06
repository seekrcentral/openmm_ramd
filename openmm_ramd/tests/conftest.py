"""
conftest.py

configurations for openmm_ramd tests
"""

import os
import pytest

import openmm_ramd.analyze.base as analyze_base

TEST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

test_small_log = os.path.join(TEST_DIRECTORY, "data/small_linear.txt")
test_large_log = os.path.join(TEST_DIRECTORY, "data/large_linear.txt")
test_small_log_end = os.path.join(TEST_DIRECTORY, "data/small_with_end.txt")
test_ran_outta_time = os.path.join(TEST_DIRECTORY, "data/ran_outta_time.txt")


@pytest.fixture(scope="session")
def small_linear_logfile_persistent_quick():
    """
    Create logfile output that is persistent across the tests.
    """
    endTimes, forceRAMD, timeStep \
        = analyze_base.parse_ramd_log_file_quick(test_small_log_end)
    return endTimes, forceRAMD, timeStep
