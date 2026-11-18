"""
Unit and regression test for the openmm_ramd package.
"""

# Import package, test suite, and other packages as needed
import sys
import os

try:
    import openmm
except ImportError:
    import simtk.openmm as openmm
    
try:
    import openmm.app as openmm_app
except ImportError:
    import simtk.openmm.app as openmm_app
    
try:
    import openmm.unit as unit
except ImportError:
    import simtk.openmm.unit as unit

import openmm_ramd.openmm_ramd as openmm_ramd
import openmm_ramd.integrator as ramd_integrator

TEST_DIRECTORY = os.path.dirname(__file__)
ROOT_DIRECTORY = os.path.join(TEST_DIRECTORY, "..")

def make_openmm_ramd_simulation_object(
        log_file_name=None, ramd_steps=10, r_min_ramd=0.025, force_out_freq=10, 
        max_dist=12.0 * unit.angstroms):
    time_step = 0.002 * unit.picoseconds
    temperature = 298.15 * unit.kelvin
    ramd_force_magnitude = 14.0 * unit.kilocalories_per_mole / unit.angstrom
    rec_indices = [569, 583, 605, 617, 1266, 1292, 1299, 1374, 1440, 1459, 1499,
                   1849, 1872, 1892, 2256, 2295, 2352, 2557]
    lig_indices = [3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 
                   3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278,
                   3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288]
    prmtop_filename = os.path.join(TEST_DIRECTORY, "../data/hsp90_INH.prmtop")
    input_pdb_file = os.path.join(TEST_DIRECTORY, "../data/hsp90_INH.pdb")
    prmtop = openmm_app.AmberPrmtopFile(prmtop_filename)
    mypdb = openmm_app.PDBFile(input_pdb_file)
    box_vectors = (mypdb.topology).getPeriodicBoxVectors()

    assert box_vectors is not None, "No box vectors "\
        "found in {}. ".format(input_pdb_file) \
        + "Box vectors for an anchor must be defined with a CRYST "\
        "line within the PDB file."
    
    system = prmtop.createSystem(
        nonbondedMethod=openmm_app.PME, nonbondedCutoff=1.0*unit.nanometers,
        constraints=openmm_app.HBonds)
    
    #integrator = openmm.LangevinIntegrator(
    #    temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    integrator = ramd_integrator.RamdLangevinIntegrator(
        temperature, 1/unit.picosecond, time_step, lig_indices, rec_indices, 
        mypdb.positions)
    simulation = openmm_ramd.RAMDSimulation(
        prmtop.topology, system, integrator, ramd_force_magnitude, lig_indices, 
        rec_indices, ramd_steps=ramd_steps, r_min_ramd=r_min_ramd, 
        force_out_freq=force_out_freq, max_dist=max_dist,
        log_file_name=log_file_name, max_travel=9e9*unit.angstroms)
    
    simulation.context.setPositions(mypdb.positions)
    simulation.context.setPeriodicBoxVectors(*box_vectors)
        
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(
        openmm_app.StateDataReporter(
            sys.stdout, 10, step=True,
            potentialEnergy=True, temperature=True, volume=True))
    return simulation

def test_openmm_ramd_simulation():
    simulation = make_openmm_ramd_simulation_object()
    simulation.step(50)
    simulation.recompute_RAMD_force()
    simulation.step(50)
    return

def test_openmm_run_RAMD_sim(tmp_path):
    log_file_name = os.path.join(tmp_path, "ramd.log")
    simulation = make_openmm_ramd_simulation_object(
        log_file_name, ramd_steps=10, r_min_ramd=0.025*unit.angstroms, 
        force_out_freq=10, max_dist=12.0*unit.angstroms)
    step_counter = simulation.run_RAMD_sim(max_num_steps=50)
    assert step_counter == 50
    assert os.path.exists(log_file_name)
    
