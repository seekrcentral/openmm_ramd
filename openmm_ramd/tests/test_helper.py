"""
test_helper.py

Tests for the helper module.
"""

import json
import os
import sys

import pytest
import openmm.app as openmm_app

TEST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
EXAMPLES_DIRECTORY = os.path.join(TEST_DIRECTORY, "..", "examples")
EXAMPLE_PARAMS = os.path.join(EXAMPLES_DIRECTORY, "example_params.json")

sys.path.insert(0, EXAMPLES_DIRECTORY)
import run_swarm
import run_swarm_iterative


def short_example_params(tmp_path):
    with open(EXAMPLE_PARAMS) as f:
        parameters = json.load(f)
    ff_parameters = parameters["ff_parameters"]
    for key in ("prmtop_filename", "pdb_filename"):
        ff_parameters[key] = os.path.abspath(
            os.path.join(EXAMPLES_DIRECTORY, ff_parameters[key]))
    ramd_parameters = parameters["ramd_parameters"]
    ramd_parameters["num_starting_structures"] = 1
    ramd_parameters["num_velocity_restarts"] = 2
    ramd_parameters["steps_between_structures"] = 1
    ramd_parameters["steps_per_RAMD_update"] = 1
    ramd_parameters["position_print_freq"] = 1
    ramd_parameters["max_num_steps"] = 2
    ramd_parameters["RAMD_max_distance"] = 1e-6
    parameter_file = os.path.join(tmp_path, "example_params.json")
    with open(parameter_file, "w") as f:
        json.dump(parameters, f)
    return parameter_file


def skip_minimization(monkeypatch):
    monkeypatch.setattr(
        openmm_app.Simulation, "minimizeEnergy",
        lambda self, *args, **kwargs: None)


@pytest.mark.needs_cuda
def test_run_swarm(tmp_path, monkeypatch):
    skip_minimization(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run_swarm.run_swarm(short_example_params(tmp_path), cuda_device_index="0")
    assert os.path.exists("starting_0.pdb")
    assert os.path.exists("ramd_0_0.log")
    assert os.path.exists("ramd_0_1.log")
    assert os.path.exists("ramd_frames_0_0.dcd")


@pytest.mark.needs_cuda
def test_run_swarm_iterative(tmp_path, monkeypatch):
    skip_minimization(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run_swarm_iterative.run_swarm_iterative(
        short_example_params(tmp_path), cuda_device_index="0")
    assert os.path.exists("starting_0.pdb")
    assert os.path.exists("starting_0.xml")
    assert os.path.exists("ramd_0_0.log")
    assert os.path.exists("ramd_0_1.log")
    assert not os.path.exists("ramd_0_2.log")
    assert not os.path.exists("starting_1.pdb")
