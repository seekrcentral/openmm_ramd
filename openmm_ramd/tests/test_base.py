"""
test_base.py

Tests for the base module.
"""

import openmm_ramd.analyze.base as base

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