"""
This sample script provides a template that one can use to run their own
RAMD simulations.
"""

import run_amber_RAMD

prmtop_filename = "../data/hsp90_INH.prmtop"
input_pdb_file = "../data/hsp90_INH.pdb"

# Output equilibration trajectory
trajectory_filename = "ramd_trajectory.dcd"

# The interval between updates to the equilibration trajectory
steps_per_trajectory_update = 50000

# Whether to minimize
minimize = True

# The max number of RAMD steps to take
num_steps = 10000000 # 20 nanoseconds

# The interval between energy printed to standard output
steps_per_energy_update = 300000

# time step of simulation 
time_step = 0.002

# Enter the atom indices whose center of mass defines the receptor binding site
rec_indices = [569, 583, 605, 617, 1266, 1292, 1299, 1374, 1440, 1459, 1499,
               1849, 1872, 1892, 2256, 2295, 2352, 2557]

# Indices for VMD selection
# 569 583 605 617 1266 1292 1299 1374 1440 1459 1499 1849 1872 1892 2256 2295 2352 2557

restraint_indices = rec_indices
                
# Enter the atom indices of the ligand molecule
lig_indices = [3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 
               3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278,
               3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288]

# Indices for VMD selection
# 3259 3260 3261 3262 3263 3264 3265 3266 3267 3268 3269 3270 3271 3272 3273 3274 3275 3276 3277 3278 3279 3280 3281 3282 3283 3284 3285 3286 3287 3288

# The RAMD force constant
ramd_force_magnitude = 14.0

# simulation initial and target temperature
temperature = 298.15

# If constant pressure is desired
target_pressure = 1.0

# Define which GPU to use
cuda_index = "0"

# Nonbonded cutoff
nonbonded_cutoff = 1.0

# The interval between RAMD force evaluations and updates
steps_per_RAMD_update = 50

RAMD_cutoff_distance = 0.0025

RAMD_max_distance = 1.5

ramd_log_filename = "ramd.log"

run_amber_RAMD.run_amber_RAMD(
    input_pdb_file, prmtop_filename, rec_indices, lig_indices,
    cuda_device_index=cuda_index, temperature=temperature, 
    nonbonded_cutoff=nonbonded_cutoff, target_pressure=target_pressure, 
    time_step=time_step, ramd_force_magnitude=ramd_force_magnitude, 
    steps_per_RAMD_update=steps_per_RAMD_update,
    RAMD_cutoff_distance=RAMD_cutoff_distance, 
    RAMD_max_distance=RAMD_max_distance,
    ramd_log_filename=ramd_log_filename, minimize=minimize,
    steps_per_energy_update=steps_per_energy_update, 
    steps_per_trajectory_update=steps_per_trajectory_update, 
    trajectory_filename=trajectory_filename,
    max_num_steps=num_steps, restraint_indices=restraint_indices)
