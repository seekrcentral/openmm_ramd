openmm_ramd
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/openmm_ramd/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/openmm_ramd/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/openmm_ramd/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/openmm_ramd/branch/master)


Implement random accelerated molecular dynamics (RAMD) in OpenMM.

WARNING: This program is in the early stages and may not perform correctly.
Use at your own discretion.

## Overview

Random accelerated molecular dynamics (RAMD) was developed to accelerate
ligand exit from a binding pocket of a receptor. One can read more about
the technique here:

https://kbbox.h-its.org/toolbox/methods/molecular-simulation/random-acceleration-molecular-dynamics-ramd/

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
Many of the dependencies for OpenMM RAMD will be installed alongside SEEKR2, but
some must be installed separately, and are installed first, before OpenMM RAMD.

#### OpenMM

OpenMM is required for the molecular dynamics (MD).

The easiest, quickest way to install the OpenMM is to use
Conda. If you don't already have Conda installed, Download Conda with 
Python version 3.8 from 
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
python setup.py install
```

If you get an error stating “No module named ‘Cython’”, this can usually be 
remedied by installing/updating Cython with:

```
pip install --upgrade cython
```


### Testing OpenMM RAMD (Optional)
To test OpenMM RAMD, run the following command in the openmm_ramd/ directory:

```
python setup.py test
```

## Run

OpenMM RAMD is intended to be run as an API, not as a standalone program,
therefore a number of example python scripts have been provided in 
openmm_ramd/openmm_ramd/examples. From within that directory one may try,
for example, the following script:

python hsp90_ramd_example.py

One may copy and adapt this script to their own systems and settings.

### Copyright

Copyright (c) 2021, Lane Votapka


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
