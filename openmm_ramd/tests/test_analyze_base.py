"""
test_analyze_base.py

Tests for the analyze.base module.
"""

import os

import openmm_ramd.analyze.base as base

TEST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

test_small_log = os.path.join(TEST_DIRECTORY, "data/small_linear.txt")
test_large_log = os.path.join(TEST_DIRECTORY, "data/large_linear.txt")
test_small_log_end = os.path.join(TEST_DIRECTORY, "data/small_with_end.txt")
test_ran_outta_time = os.path.join(TEST_DIRECTORY, "data/ran_outta_time.txt")

def test_pretty_string_value_error():
    """
    Test the function to print 'pretty' scientific notation to the
    terminal.
    """
    mystr = base.pretty_string_value_error(
        5.6e-2, 2.0e-3, error_digits=1, use_unicode=False)
    expectedstr = "(5.6 +/- 0.2) * 10^-02"
    assert(mystr == expectedstr)
    mystr = base.pretty_string_value_error(
        5.6e-2, 2.0e-1, error_digits=1, use_unicode=False)
    expectedstr = "(5.6 +/- 20.0) * 10^-02"
    assert(mystr == expectedstr)
    mystr = base.pretty_string_value_error(
        1.23456789e8, 4.5678e5, error_digits=2, use_unicode=False)
    expectedstr = "(1.2346 +/- 0.0046) * 10^+08"
    assert(mystr == expectedstr)
    return

def test_parse_ramd_log_file_quick():
    """
    Test the function to parse the RAMD log file.
    """
    endTimes, forceRAMD, timeStep \
        = base.parse_ramd_log_file_quick(test_small_log)
    assert(endTimes == [None])
    assert(forceRAMD == 0.1)
    assert(timeStep == 0.002)
    
    endTimes, forceRAMD, timeStep \
        = base.parse_ramd_log_file_quick(test_large_log)
    assert(forceRAMD == 0.1)
    assert(timeStep == 0.002)

    endTimes, forceRAMD, timeStep \
        = base.parse_ramd_log_file_quick(test_small_log_end)
    assert(endTimes == [5612050])
    assert(forceRAMD == 0.1)
    assert(timeStep == 0.002)


def test_escaped_successfully():
    assert base.escaped_successfully(test_small_log) == False
    assert base.escaped_successfully(test_large_log) == False
    assert base.escaped_successfully(test_small_log_end) == True
    assert base.escaped_successfully(test_ran_outta_time) == False
    return

def test_ran_out_of_time():
    assert base.ran_out_of_time(test_small_log) == False
    assert base.ran_out_of_time(test_large_log) == False
    assert base.ran_out_of_time(test_small_log_end) == False
    assert base.ran_out_of_time(test_ran_outta_time) == True
    return