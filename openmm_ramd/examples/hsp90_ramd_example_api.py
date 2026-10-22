"""
This sample script provides a template that one can use to run their own
RAMD simulations.
"""

import time
from sys import stdout

import openmm.app as app
import openmm as mm
import openmm.unit as unit

from openmm_ramd import openmm_ramd
import openmm_ramd.integrator as ramd_integrator

prmtop_filename = "../data/hsp90_INH.prmtop"
input_pdb_file = "../data/hsp90_INH.pdb"

# Output equilibration trajectory
trajectory_filename = "ramd_trajectory.dcd"

# The interval between updates to the equilibration trajectory
steps_per_trajectory_update = 50000

# Whether to minimize
minimize = True

# The total number of RAMD steps to take
num_steps = 10000000 # 20 nanoseconds

# The interval between energy printed to standard output
steps_per_energy_update = 50000

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

RAMD_max_distance = 3.0 * unit.nanometer

prmtop = app.AmberPrmtopFile(prmtop_filename)
mypdb = app.PDBFile(input_pdb_file)
box_vectors = (mypdb.topology).getPeriodicBoxVectors()

assert box_vectors is not None, "No box vectors "\
    "found in {}. ".format(input_pdb_file) \
    + "Box vectors for an anchor must be defined with a CRYST "\
    "line within the PDB file."

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=nonbonded_cutoff,
        constraints=app.HBonds)
if constant_pressure:
    barostat = mm.MonteCarloBarostat(target_pressure, temperature, 25)
    system.addForce(barostat)
    
integrator = ramd_integrator.RamdLangevinIntegrator(
        temperature, 1/unit.picosecond, time_step, lig_indices, rec_indices, 
        mypdb.positions)
platform = mm.Platform.getPlatformByName('CUDA')
properties = {"CudaDeviceIndex": cuda_index, "CudaPrecision": "mixed"}

simulation = openmm_ramd.RAMDSimulation(
    prmtop.topology, system, integrator, ramd_force_magnitude, lig_indices, 
    rec_indices, ramd_steps=steps_per_RAMD_update,
    r_min_ramd=RAMD_cutoff_distance,
    force_out_freq=steps_per_RAMD_update,
    max_dist=RAMD_max_distance, max_travel=9e9 * unit.nanometer,
    platform=platform, properties=properties, log_file_name="ramd.log",
    ramp_start_ns=0.0,
    ramp_force_increase_per_ns=0.0 * unit.kilocalories_per_mole / unit.angstrom)

simulation.context.setPositions(mypdb.positions)
simulation.context.setPeriodicBoxVectors(*box_vectors)
if minimize:
    simulation.minimizeEnergy()
    
simulation.context.setVelocitiesToTemperature(temperature)
simulation.reporters.append(app.StateDataReporter(stdout, steps_per_energy_update, step=True,
            potentialEnergy=True, temperature=True, volume=True))
simulation.reporters.append(app.DCDReporter(
    trajectory_filename, steps_per_trajectory_update, enforcePeriodicBox=True))

start_time = time.time()
counter = simulation.run_RAMD_sim(max_num_steps=num_steps)

total_time = time.time() - start_time
simulation_in_ns = counter * time_step.value_in_unit(unit.picoseconds) * 1e-3
total_time_in_days = total_time / (86400.0)
ns_per_day = simulation_in_ns / total_time_in_days
print("RAMD benchmark:", ns_per_day, "ns/day")

