"""
Unit and regression test for the openmm_ramd package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import openmm_ramd


def test_openmm_ramd_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "openmm_ramd" in sys.modules
