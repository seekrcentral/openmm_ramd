"""
This sample script provides a template that one can use to run their own
RAMD simulations.
"""

import time
from sys import stdout

import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as unit
import parmed
import mdtraj
import numpy as np

import openmm_ramd.base as base
from openmm_ramd import openmm_ramd

prmtop_filename = "../data/hsp90_INH.prmtop"
input_pdb_file = "../data/hsp90_INH.pdb"

# Output equilibration trajectory
trajectory_filename = "ramd_trajectory.pdb"

# The interval between updates to the equilibration trajectory
steps_per_trajectory_update = 50000

# Whether to minimize
minimize = True

# The total number of RAMD steps to take
num_steps = 1000000 # 2 nanoseconds

# The interval between energy printed to standard output
steps_per_energy_update = 300000

# time step of simulation 
time_step = 0.002 * unit.picoseconds

# Enter the atom indices whose center of mass defines the receptor binding site
rec_indices = [569, 583, 605, 617, 1266, 1292, 1299, 1374, 1440, 1459, 1499,
               1849, 1872, 1892, 2256, 2295, 2352, 2557]

# Indices for VMD selection
# 569 583 605 617 1266 1292 1299 1374 1440 1459 1499 1849 1872 1892 2256 2295 2352 2557
                
# Enter the atom indices of the ligand molecule
lig_indices = [3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 
               3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278,
               3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288]

# Indices for VMD selection
# 3259 3260 3261 3262 3263 3264 3265 3266 3267 3268 3269 3270 3271 3272 3273 3274 3275 3276 3277 3278 3279 3280 3281 3282 3283 3284 3285 3286 3287 3288

# To hold the ligand in place during the equilibration, a harmonic force 
#  keeps the center of mass of the ligand and binding site at a constant
#  distance
ramd_force_magnitude = 14.0 * unit.kilocalories_per_mole / unit.angstrom

# simulation initial and target temperature
temperature = 298.15 * unit.kelvin

# If constant pressure is desired
constant_pressure = True
target_pressure = 1.0 * unit.bar

# Define which GPU to use
cuda_index = "0"

# Nonbonded cutoff
nonbonded_cutoff = 1.0 * unit.nanometer

# The interval between RAMD force evaluations and updates
steps_per_RAMD_update = 50

RAMD_cutoff_distance = 0.0025 * unit.nanometer

RAMD_max_distance = 1.5 * unit.nanometer

ramd_log_filename = "ramd.log"

#starting_ligand_site_distance = get_site_ligand_distance(
#    input_pdb_file, rec_indices, lig_indices)
#print("Starting ligand-site distance:", starting_ligand_site_distance)

# Modify target_distance if you want the ligand to be pulled to a different
# distance. For example:
# target_distance = 0.6 * unit.nanometers
#target_distance = starting_ligand_site_distance

########################################################
# DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
########################################################
prmtop = app.AmberPrmtopFile(prmtop_filename)
mypdb = app.PDBFile(input_pdb_file)
pdb_parmed = parmed.load_file(input_pdb_file)
assert pdb_parmed.box_vectors is not None, "No box vectors "\
    "found in {}. ".format(input_pdb_file) \
    + "Box vectors for an anchor must be defined with a CRYST "\
    "line within the PDB file."

box_vectors = pdb_parmed.box_vectors

system = prmtop.createSystem(
    nonbondedMethod=app.PME, nonbondedCutoff=nonbonded_cutoff,
    constraints=app.HBonds)
if constant_pressure:
    barostat = mm.MonteCarloBarostat(target_pressure, temperature, 25)
    system.addForce(barostat)
    
integrator = mm.LangevinIntegrator(temperature, 1/unit.picosecond, time_step)
platform = mm.Platform.getPlatformByName('CUDA')
properties = {"CudaDeviceIndex": cuda_index, "CudaPrecision": "mixed"}

#simulation = app.Simulation(prmtop.topology, system, integrator, platform, properties)
simulation = openmm_ramd.RAMDSimulation(
    prmtop.topology, system, integrator, ramd_force_magnitude, lig_indices, 
    rec_indices, ramdSteps=steps_per_RAMD_update, 
    rMinRamd=RAMD_cutoff_distance.value_in_unit(unit.angstroms), 
    forceOutFreq=steps_per_RAMD_update, 
    maxDist=RAMD_max_distance.value_in_unit(unit.angstrom),
    platform=platform, properties=properties, log_file_name=ramd_log_filename)

simulation.context.setPositions(mypdb.positions)
simulation.context.setPeriodicBoxVectors(*box_vectors)
if minimize:
    simulation.minimizeEnergy()
    
simulation.context.setVelocitiesToTemperature(temperature)
simulation.reporters.append(app.StateDataReporter(stdout, steps_per_energy_update, step=True,
            potentialEnergy=True, temperature=True, volume=True))
pdb_reporter = app.PDBReporter(trajectory_filename, steps_per_trajectory_update)
simulation.reporters.append(pdb_reporter)
start_time = time.time()
step_counter = simulation.run_RAMD_sim(max_num_steps=num_steps)

total_time = time.time() - start_time
simulation_in_ns = step_counter * time_step.value_in_unit(unit.picoseconds) * 1e-3
total_time_in_days = total_time / (86400.0)
ns_per_day = simulation_in_ns / total_time_in_days
print("RAMD benchmark:", ns_per_day, "ns/day")

#end_distance = get_site_ligand_distance(output_pdb_file, rec_indices, 
#                                                         lig_indices)
#print("Final ligand-site distance:", end_distance)
