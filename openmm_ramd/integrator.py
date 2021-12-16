"""
integrator.py

Set the customRamdIntegrator and other integrator subclasses.
"""

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
    
kcal_per_mole_per_angstrom = unit.kilocalories_per_mole / unit.angstrom

"""
class CustomRamdIntegrator(openmm.CustomIntegrator):
    ""
    customRamdIntegrator is a subclass that will be inherited in 
    order to implement the random forces, evaluation of movement,
    and logging for a RAMD simulation.
    ""
    
    def __init__(self, ramd_force_magnitude=0.0*kcal_per_mole_per_angstrom):
        self.ramd_force_magnitude = ramd_force_magnitude
        return
    
    def setRAMDForceMagnitude(self, ramd_force_magnitude):
        self.ramd_force_magnitude = ramd_force_magnitude
        return
    
    def getRAMDForceMagnitude(self):
        return self.ramd_force_magnitude
    
    def initializeRAMDVariablesInstructions(self):
        ""
        Initialize the variables used by the integrator in the RAMD 
        simulation.
        ""
        self.addGlobalVariable("ramd_force_magnitude", self.ramd_force_magnitude)
        self.addGlobalVariable("gauss_x", 0)
        self.addGlobalVariable("gauss_y", 0)
        self.addGlobalVariable("gauss_z", 0)
        self.addGlobalVariable("gauss_len", 0)
        self.addGlobalVariable("fx", 0)
        self.addGlobalVariable("fy", 0)
        self.addGlobalVariable("fz", 0)
        
        return
    
    def recomputeRandomForceInstructions(self):
        ""
        Use a normalized 3D gaussian to compute the random force vector
        ""
        
        self.addComputeGlobal("gauss_x", "gaussian")
        self.addComputeGlobal("gauss_y", "gaussian")
        self.addComputeGlobal("gauss_z", "gaussian")
        self.addComputeGlobal("gauss_len", "sqrt(gauss_x*gauss_x + "\
                              +"gauss_y*gauss_y + gauss_z*gauss_z)")
        self.addComputeGlobal("fx", "ramd_force_magnitude * gauss_x / gauss_len")
        self.addComputeGlobal("fy", "ramd_force_magnitude * gauss_y / gauss_len")
        self.addComputeGlobal("fz", "ramd_force_magnitude * gauss_z / gauss_len")
        
        return
"""