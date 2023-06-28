"""
Utilities to write RAMD output to a file.
"""

RAMD_PREFIX = "RAMD: "
RAMD_FORCE_PREFIX = "RAMD FORCE: "
EXIT_PREFIX = "EXIT: "

class RAMD_logger():
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "w")
        self.write_header()
    
    def __del__(self):
        self.file.close()
    
    def write_header(self):
        self.log("")
        self.log("  -------------------------------------------------------------------  ")
        self.log("  Random Acceleration Molecular Dynamics Simulation using openmm_ramd")
        self.log("  -------------------------------------------------------------------  ")
        self.log("")
        return
    
    def log(self, string, prefix=RAMD_PREFIX, print_also=False):
        full_string = prefix + string + "\n"
        self.file.write(full_string)
        if print_also:
            print(full_string)
        return
    
    def exit_log(self, string, print_also=False):
        self.log(string, prefix=EXIT_PREFIX, print_also=print_also)
        return
    
    def timestep_log(self, string, timestep):
        new_string = "{} {}".format(timestep, string)
        self.log(new_string)
        return
    
    def force_log(self, string, timestep):
        new_string = "{} {}".format(timestep, string)
        self.log(new_string, prefix=RAMD_FORCE_PREFIX)
        return