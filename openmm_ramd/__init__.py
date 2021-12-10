"""Implement random accelerated molecular dynamics (RAMD) in OpenMM."""

# Add imports here
from .openmm_ramd import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
