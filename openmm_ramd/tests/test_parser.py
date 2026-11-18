"""
Test openmm_ramd parser.py functions.
"""
import os

TEST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
def test_parse_ramd_log_file_quick(small_linear_logfile_persistent_quick):
    [endTimes, forceRAMD, timeStep] = small_linear_logfile_persistent_quick
    assert endTimes == [5612050]
    assert forceRAMD == 0.1
    assert timeStep == 0.002
    return
