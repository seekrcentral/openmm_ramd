"""
A base module for analyzing RAMD simulations.
"""

import re

import numpy as np

def parse_ramd_log_file_quick(ramd_log_filename, include_all_times=False):
    """
    Parse less information out of the RAMD log files, namely some of the
    constants, as well as the ending times of the simulations. This information
    is all that's needed for tau-ramd as well as Maximova's method.
    """
    end_times = []
    forceRAMD = None
    timeStep = None
    num_simulations = 0
    
    found_constants = False
    
    # A value of -1 indicates that no exit events happened
    exit_step = None
    
    with open(ramd_log_filename, "rb") as f:
        for i, bline in enumerate(f):
            try:
                line = bline.decode("utf-8")
            except UnicodeDecodeError:
                continue
            forceRAMD_search = re.match(r"RAMD: forceRAMD * (\d+\.?\d*[eE]?[+-]?\d*)", line)
            if forceRAMD_search:
                forceRAMD_ = float(forceRAMD_search.group(1))
                if forceRAMD is not None:
                    assert forceRAMD == forceRAMD_, \
                        "The log files may not have conflicting values of "\
                        "forceRAMD"
                    # Also the start of a new simulation
                    num_simulations += 1
                    end_times.append(exit_step)
                    exit_step = None
                    
                else:
                    forceRAMD = forceRAMD_
            
            timeStep_search = re.match(r"RAMD: timeStep * (\d+\.?\d*[eE]?[+-]?\d*)", line)
            if timeStep_search:
                timeStep_ = float(timeStep_search.group(1))
                if timeStep is not None:
                    assert timeStep == timeStep_, \
                        "The log files may not have conflicting values of "\
                        "timeStep"
                else:
                    timeStep = timeStep_
            
            if include_all_times:
                ramd_force_search = re.match(r"RAMD FORCE: (\d+) >", line)
                if ramd_force_search:
                    exit_step = int(ramd_force_search.group(1))
            
            exit_event_search = re.match(r"EXIT: (\d+)  > MAX DISTANCE LIGAND COM - PROTEIN COM REACHED", line)
            if exit_event_search:
                exit_step = int(exit_event_search.group(1))
            
    end_times.append(exit_step)
    assert forceRAMD is not None, \
        "The variable forceRAMD was not assigned in the log file."
    assert timeStep is not None, \
        "The variable timeStep was not assigned in the log file."
    
    return end_times, forceRAMD, timeStep

def pretty_string_value_error(value, error, error_digits=2, use_unicode=True):
    """
    Copied from seekr2
    
    Returns a value/error combination of numbers in a scientifically 
    'pretty' format.
    
    Scientific quantities often come as a *value* (the actual 
    quantity) and the *error* (the uncertainty in the value).
        
    Given two floats, value and error, return the two in a 
    'pretty' formatted string: where the value and error are truncated
    at the correct precision.
    
    Parameters
    ----------
    value : float
        The quantity in question
        
    error : float
        The uncertainty of the quantity
        
    error_digits : int, default 2
        How many significant figures the error has. Scientific 
        convention holds that errors have 1 or (at most) 2 significant
        figures. The larger number of digits is chosen here by default.
        
    Returns
    -------
    new_string : str
        A new list of strings sorted numerically
    
    Examples
    --------
    
        >>> pretty_string_value_error(1.23456789e8, 4.5678e5, 
                                      error_digits=2)
        "1.2346 +/- 0.0046 * 10^+08"
    
        >>> pretty_string_value_error(5.6e-2, 2.0e-3, error_digits=1)
        "5.6 +/- 0.2 * 10^-02"
    """
    if value is None:
        return "None"
    if error is None or not np.isfinite(error):
        if use_unicode:
            new_string = "{:.6E} \u00B1 UNKNOWN ERROR MARGIN".format(value)
        else:
            new_string = "{:.6E} +/- UNKNOWN ERROR MARGIN".format(value)
    else:
        if not np.isfinite(value):
            return str(value)
        assert "e" in "{:e}".format(value), "Cannot convert into scientific "\
            "notation: {1}".format(value)
        value_mantissa_str, value_exponent_str = \
            "{:e}".format(value).strip().split('e')
        value_mantissa = float(value_mantissa_str)
        value_exponent = int(value_exponent_str)
        error_mantissa_str, error_exponent_str = \
            "{:e}".format(error).strip().split('e')
        error_mantissa = float(error_mantissa_str)
        error_exponent = int(error_exponent_str)
        padding = value_exponent - error_exponent + error_digits - 1
        if padding < 1: padding = 1
        exp_diff = error_exponent - value_exponent
        string_for_formatting = "{:.%df}" % padding
        new_value_mantissa = string_for_formatting.format(value_mantissa)
        new_error_mantissa = string_for_formatting.format(
            error_mantissa*10**exp_diff)
        if use_unicode:
            new_string = "(%s \u00B1 %s) * 10^%s" % (
                new_value_mantissa, new_error_mantissa, value_exponent_str)
        else:
            new_string = "(%s +/- %s) * 10^%s" % (
                new_value_mantissa, new_error_mantissa, value_exponent_str)
    return new_string

def get_last_lines_in_file(filename):
    """
    Retrieve the last 
    """
    with open(filename, "rb") as f:
        f.seek (0, 2)           # Seek @ EOF
        fsize = f.tell()        # Get Size
        f.seek (max (fsize-2048, 0), 0) # Set pos @ last n chars
        lines = []
        for i, bline in enumerate(f):
            try:
                line = bline.decode("utf-8")
            except UnicodeDecodeError:
                continue
            
            lines.append(line)
    return lines

def escaped_successfully(log_filename):
    last_lines = get_last_lines_in_file(log_filename)
    for i, bline in enumerate(last_lines):
        try:
            line = bline.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            line = bline
        
        exit_event_search = re.match(r"EXIT: (\d+)  > MAX DISTANCE LIGAND COM - PROTEIN COM REACHED", line)
        if exit_event_search:
            return True
        
    return False
        
def ran_out_of_time(log_filename):
    last_lines = get_last_lines_in_file(log_filename)
    for i, line in enumerate(last_lines):
        exit_event_search = re.match(r"EXIT: (\d+)  > MAXIMUM TIME EXCEEDED", line)
        if exit_event_search:
            return True
        
    return False
