openmm_ramd
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/openmm_ramd/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/openmm_ramd/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/openmm_ramd/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/openmm_ramd/branch/master)


Implement random acceleration molecular dynamics (RAMD) in OpenMM.

## Overview

Random acceleration molecular dynamics (RAMD) was developed to accelerate
ligand exit from a binding pocket of a receptor. One can read more about
the technique here:

https://kbbox.h-its.org/toolbox/methods/molecular-simulation/random-acceleration-molecular-dynamics-ramd/

<img width="800" height="600" alt="cmp_1_exit_arrow_last5s" src="https://github.com/user-attachments/assets/33393202-bd9b-49b4-90d5-513cf923bb90" />


This package allows one to run RAMD using OpenMM, optionally generating
an output log file that resemble the output produced by the NAMD version 
of RAMD.

RAMD may be used to generate starting structures along the unbinding
pathway, in particular, for the SEEKR2 program. Please see the following
links to obtain information about SEEKR2 and how RAMD is used with it:

https://github.com/seekrcentral/seekr2.git
https://github.com/seekrcentral/seekrtools.git

In particular, the Seekrtools program HIDR has an option to use RAMD to 
generate starting structures for SEEKR using the RAMD method.

## Quick Install

### Dependencies


#### OpenMM

OpenMM is required for the molecular dynamics (MD).

The easiest, quickest way to install the OpenMM is to use
Conda. If you don't already have Conda installed, Download Conda with 
Python version 3.10 from 
https://conda.io/en/latest/miniconda.html and run the downloaded script and 
fill out the prompts. 

With Conda working, install OpenMM:

```
conda install -c conda-forge openmm
```

### Install OpenMM RAMD

Once the dependencies are installed, we may install OpenMM RAMD. First, clone 
this repository and install the package:

```
git clone https://github.com/seekrcentral/openmm_ramd.git
cd openmm_ramd
python -m pip install .
```

### Testing OpenMM RAMD (Optional)
To test OpenMM RAMD, run the following command in the openmm_ramd/ directory:

```
pytest
```

## Run

There are a number of ways to run RAMD with this package. The following scripts can be found in openmm_ramd/openmm_ramd/examples

To run a swarm of trajectories, like for tauRAMD ranking, the easiest way is to construct a JSON file with the parameters (see example_params.json) and run run_swarm.py:
```
python run_swarm.py example_params.json
```
This runs a fixed amount of replicas and trajectories, defined in the parameters.

To replicate the iterative procedure used in most tau-RAMD publications, one can use run_swarm_iterative.py, which will add replicas (up to a maximum of 8) and trajectories when the standard deviation exceeds a given threshold:

```
python run_swarm_iterative.py example_params.json
```

Alternatively, one might want to run a single RAMD simulation, which can be done with a Python script by using the functionality of run_amber_RAMD.py:
```
python hsp90_ramd_example.py
```
or from the command line:
```
bash hsp90_ramd_example.bash
```

For advanced usage, you can directly access the RAMD simulation and integrators within a Python script, giving you complete control over the simulation as you would in a standard OpenMM script:

```
python hsp90_ramd_example_api.py
```

## Parameters
RAMD parameters for the usage examples above fall into three categories. These parameters can be defined in the parameters.json for swarm workflows, from the command line, or in Python scripts.

##### Forcefield parameters

<table>
  <tr><td><code>type</code></td><td>can be <code>amber</code> or <code>openmm_system_xml</code></td></tr>
  <tr><td><code>prmtop_filename</code></td><td>if <code>type = amber</code>, a .prmtop file provides the topology and parameters</td></tr>
  <tr><td><code>system_xml_filename</code></td><td>if <code>type = openmm_system_xml</code>, this provides the serialized OpenMM system with forces and particle definitions. Note that this is not the same as the simulation state XMLs used in the iterative workflow.</td></tr>
  <tr><td><code>pdb_filename</code></td><td>both types require a PDB file for coordinates and periodic box vectors.</td></tr>
</table>

##### Simulation parameters

<table>
  <tr><td><code>temperature</code></td><td>temperature in Kelvin.</td></tr>
  <tr><td><code>time_step</code></td><td>integration timestep in picoseconds.</td></tr>
  <tr><td><code>integrator_type</code></td><td>RAMD integrator to use. Available options are <code>langevin, brownian, verlet, nose_hoover</code></td></tr>
  <tr><td><code>nonbonded_cutoff</code></td><td>nonbonded cutoff in nanometers, default 0.8</td></tr>
  <tr><td><code>pressure</code></td><td>Pressure in bar for NPT simulation using a Monte Carlo barostat, default null.</td></tr>
  <tr><td><code>hydrogen_mass</code></td><td>Hydrogen mass in amu for HMR. Default is null, which uses original masses.</td></tr>
  <tr><td><code>friction_coeff</code></td><td>Langevin friction coefficient in ps^-1, default 1.0</td></tr>
  <tr><td><code>collision_frequency</code></td><td>Nose-Hoover coupling in ps^-1, default 1.0</td></tr>
  <tr><td><code>chain_length</code></td><td>Nose-Hoover chain length, default 3.</td></tr>
  <tr><td><code>num_mts</code></td><td>number of substeps for Nose-Hoover multiple time stepping, default 3.</td></tr>
  <tr><td><code>num_yoshida_suzuki</code></td><td>number of Nose-Hoover Yoshida-Suzuki integration terms, default 7.</td></tr>
  <tr><td><code>random_seed</code></td><td>Seed for velocity initialization when not using XML restarts. Note that the RAMD force is not currently controllable by a random seed.</td></tr>
  <tr><td><code>restraint_indices</code></td><td>Indices to restrain. Default null sets no restraints.</td></tr>
  <tr><td><code>restraint_force_constant</code></td><td>Harmonic restraint force constant in kJ/mol/nm^2.</td></tr>
  <tr><td><code>steps_per_energy_update</code></td><td>steps between printing potential energy, temperature, and volume.</td></tr>
  <tr><td><code>steps_per_trajectory_update</code></td><td>How often to save frames to a DCD trajectory. Note that RAMD already outputs the first and last frames by default. Also note that saving frames creates lots of data for swarms, so use with caution.</td></tr>
</table>

##### RAMD parameters

<table>
  <tr><td><code>ligand_indices</code></td><td>Indices of the ligand atoms, which define the center of mass and which the RAMD force is applied to.</td></tr>
  <tr><td><code>receptor_indices</code></td><td>Indices of the receptor atoms, used for computing center of mass motion.</td></tr>
  <tr><td><code>ramd_force_magnitude</code></td><td>Magnitude of the RAMD force in kcal/molA</td></tr>
  <tr><td><code>steps_per_RAMD_update</code></td><td>Steps between com distance evaluations</td></tr>
  <tr><td><code>position_print_freq</code></td><td>Steps between writing center of mass positions to the log. For most use cases this should be null, which uses steps_per_RAMD_update, but some users may desire more frequent COM logging.</td></tr>
  <tr><td><code>RAMD_cutoff_distance</code></td><td>Threshold distance that ligand must have moved (in nanometers) within the <code>steps_per_RAMD_update</code> interval to retain its force direction. If ligand moves less than this distance, the force is reoriented.</td></tr>
  <tr><td><code>max_travel</code></td><td>Maximum ligand displacement (in nanometers) before force is reoriented. Set to an arbitrary large value to disable this, which is what most use cases will need.</td></tr>
  <tr><td><code>max_num_steps</code></td><td>Maximum number of steps for each RAMD trajectory.</td></tr>
  <tr><td><code>ramp_start_ns</code></td><td>Time in nanoseconds before force ramp begins.</td></tr>
  <tr><td><code>ramp_force_increase_per_ns</code></td><td>Force increase in (kcal/molA)/ns. Set to zero to disable force ramp.</td></tr>
  <tr><td><code>num_starting_structures</code></td><td>Number of replicas to generate.</td></tr>
  <tr><td><code>num_velocity_restarts</code></td><td>Number of RAMD trajectories per starting replica.</td></tr>
  <tr><td><code>steps_between_structures</code></td><td>Number of steps of conventional MD for replica generation.</td></tr>
</table>

## Analyze

RAMD simulations output logfiles which contain each trajectory's time to unbind. To integrate these times into a single estimate tau_comp, especially for ranking workflows, one can run:
```
python openmm_ramd/openmm_ramd/analyze/tau_ramd.py ramd*.log
```
from within the directory containing these logfiles. This outputs the statistics, including t_r, SD_r, SD_replicas, and tau_comp. 

## Authors and Contributors

The following people have contributed directly to the coding and validation
efforts of openmm_ramd (listed an alphabetical order of last name). 
Thanks also to everyone who has helped or will help improve this project by 
providing feedback, bug reports, or other comments.

* Rommie Amaro (principal investigator)
* Anson Noland (developer)
* Lane Votapka (lead developer)

### Copyright

Copyright (c) 2021-2026, Lane Votapka and Anson Noland


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
