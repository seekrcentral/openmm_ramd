"""
parser.py

Parse RAMD log files, and use the log files to construct trajectories of
CVs.
"""

import re
import os

import numpy as np

def add_new_frame(trajectory, n_frames, relative_COM_x, relative_COM_y, 
                  relative_COM_z, old_ligand_COM_x, old_ligand_COM_y, 
                  old_ligand_COM_z, accel_COM_x, accel_COM_y, accel_COM_z,
                  old_ligand_step, receptor_step, accel_step):
    if n_frames > 0:
        if relative_COM_x is not None:
            #assert old_ligand_step == receptor_step, \
            #    "The lines defining receptor and ligand COM are not the same "\
            #    "RAMD step."
            frame = [relative_COM_x, relative_COM_y, relative_COM_z,
                     accel_COM_x, accel_COM_y, accel_COM_z]
        else:
            frame = [old_ligand_COM_x, old_ligand_COM_y, old_ligand_COM_z,
                     accel_COM_x, accel_COM_y, accel_COM_z]
            
        assert old_ligand_step == accel_step, \
            "The lines defining ligand COM and accel vector are not the same "\
            "RAMD step."
        
        trajectory.append(frame)
        
    n_frames += 1
    return n_frames

def parse_ramd_log_file(ramd_log_filename):
    assert os.path.exists(ramd_log_filename), \
        f"File not found: {ramd_log_filename}."
    trajectories = []
    trajectory = []
    n_frames = 0
    relative_COM_x = None
    relative_COM_y = None
    relative_COM_z = None
    receptor_COM_x = None
    receptor_COM_y = None
    receptor_COM_z = None
    ligand_step = None
    ligand_COM_x = None
    ligand_COM_y = None
    ligand_COM_z = None
    old_ligand_COM_x = None
    old_ligand_COM_y = None
    old_ligand_COM_z = None
    accel_COM_x = None
    accel_COM_y = None
    accel_COM_z = None
    forceOutFreq = None
    positionPrintFreq = None
    forceRAMD = None
    timeStep = None
    temperature = None
    max_dist = None
    old_ligand_step = None
    receptor_step = None
    accel_step = None
    num_simulations = 0
        
    with open(ramd_log_filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            new_trajectory_frame = False
            new_trajectory = False
            # Determine the log output frequency
            forceOutFreq_search = re.match("RAMD: forceOutFreq * (\d+)", line)
            if forceOutFreq_search:
                forceOutFreq_ = int(forceOutFreq_search.group(1))
                if forceOutFreq is not None:
                    assert forceOutFreq == forceOutFreq_, \
                        "The log files may not have conflicting values of "\
                        "forceOutFreq"
                else:
                    forceOutFreq = forceOutFreq_
                    
            positionPrintFreq_search = re.match("RAMD: positionPrintFreq * (\d+)", line)
            if positionPrintFreq_search:
                positionPrintFreq_ = int(positionPrintFreq_search.group(1))
                if positionPrintFreq is not None:
                    assert positionPrintFreq == positionPrintFreq_, \
                        "The log files may not have conflicting values of "\
                        "positionPrintFreq"
                else:
                    positionPrintFreq = positionPrintFreq_
                
            forceRAMD_search = re.match("RAMD: forceRAMD * (\d+\.?\d*[eE]?[+-]?\d*)", line)
            if forceRAMD_search:
                forceRAMD_ = float(forceRAMD_search.group(1))
                if forceRAMD is not None:
                    assert forceRAMD == forceRAMD_, \
                        "The log files may not have conflicting values of "\
                        "forceRAMD"
                    # Also the start of a new simulation
                    num_simulations += 1
                    old_ligand_COM_x = ligand_COM_x
                    old_ligand_COM_y = ligand_COM_y
                    old_ligand_COM_z = ligand_COM_z
                    new_trajectory_frame = True
                    new_trajectory = True
                    old_ligand_step = ligand_step
                    accel_step = ligand_step
                else:
                    forceRAMD = forceRAMD_
                
            timeStep_search = re.match("RAMD: timeStep * (\d+\.?\d*[eE]?[+-]?\d*)", line)
            if timeStep_search:
                timeStep_ = float(timeStep_search.group(1))
                if timeStep is not None:
                    assert timeStep == timeStep_, \
                        "The log files may not have conflicting values of "\
                        "timeStep"
                else:
                    timeStep = timeStep_
                    
            temperature_search = re.match("RAMD: temperature * (\d+\.?\d*[eE]?[+-]?\d*)", line)
            if temperature_search:
                temperature_ = float(temperature_search.group(1))
                if temperature is not None:
                    assert temperature == temperature_, \
                        "The log files may not have conflicting values of "\
                        "temperature"
                else:
                    temperature = temperature_
                    
            max_dist_search = re.match("RAMD: maxDist * (\d+\.?\d*[eE]?[+-]?\d*)", line)
            if max_dist_search:
                max_dist_ = float(max_dist_search.group(1))
                if max_dist is not None:
                    assert max_dist == max_dist_, \
                        "The log files may not have conflicting values of "\
                        "maxDist"
                else:
                    max_dist = max_dist_
            
            # Find the lines that indicate the ligand COM
            lig_com_search = re.match("RAMD FORCE: (\d+) > LIGAND COM IS: \[ *([+-]?\d+\.\d*[eE]?[+-]?\d*) +([+-]?\d+\.\d*[eE]?[+-]?\d*) +([+-]?\d+\.\d*[eE]?[+-]?\d*) *\]", line)
            if lig_com_search:
                old_ligand_step = ligand_step
                ligand_step = int(lig_com_search.group(1))
                old_ligand_COM_x = ligand_COM_x
                old_ligand_COM_y = ligand_COM_y
                old_ligand_COM_z = ligand_COM_z
                ligand_COM_x = float(lig_com_search.group(2))
                ligand_COM_y = float(lig_com_search.group(3))
                ligand_COM_z = float(lig_com_search.group(4))
                new_trajectory_frame = True
            
            rec_com_search = re.match("RAMD FORCE: (\d+) > PROTEIN COM IS: \[ *([+-]?\d+\.\d*[eE]?[+-]?\d*) +([+-]?\d+\.\d*[eE]?[+-]?\d*) +([+-]?\d+\.\d*[eE]?[+-]?\d*) *\]", line)
            if rec_com_search:
                receptor_step = int(rec_com_search.group(1))
                receptor_COM_x = float(rec_com_search.group(2))
                receptor_COM_y = float(rec_com_search.group(3))
                receptor_COM_z = float(rec_com_search.group(4))
                relative_COM_x = ligand_COM_x - receptor_COM_x
                relative_COM_y = ligand_COM_y - receptor_COM_y
                relative_COM_z = ligand_COM_z - receptor_COM_z
                continue
            
            accel_search = re.match("RAMD: (\d+)    >>> KEEP PREVIOUS ACCELERATION DIRECTION: *\[ *([+-]?\d+\.\d*[eE]?[+-]?\d*) +([+-]?\d+\.\d*[eE]?[+-]?\d*) +([+-]?\d+\.\d*[eE]?[+-]?\d*) *\]", line)
            if accel_search:
                accel_step = int(accel_search.group(1))
                accel_COM_x = float(accel_search.group(2))
                accel_COM_y = float(accel_search.group(3))
                accel_COM_z = float(accel_search.group(4))
                continue
            
            change_accel_search = re.match("RAMD: (\d+)    >>> CHANGE ACCELERATION DIRECTION TO: *\[ *([+-]?\d+\.\d*[eE]?[+-]?\d*) +([+-]?\d+\.\d*[eE]?[+-]?\d*) +([+-]?\d+\.\d*[eE]?[+-]?\d*) *\]", line)
            if change_accel_search:
                accel_step = int(change_accel_search.group(1))
                accel_COM_x = float(change_accel_search.group(2))
                accel_COM_y = float(change_accel_search.group(3))
                accel_COM_z = float(change_accel_search.group(4))
                old_ligand_step = ligand_step
                if n_frames > 1:
                    new_trajectory_frame = True
                new_trajectory = True
            
            if new_trajectory:
                if len(trajectory) > 0:
                    trajectories.append(trajectory)
                    trajectory = []
                    n_frames = 0
                    
            if new_trajectory_frame:
                if n_frames > 0:
                    assert old_ligand_COM_x is not None, \
                        f"Error near file: {ramd_log_filename}, line: {i}"
                n_frames = add_new_frame(
                    trajectory, n_frames, relative_COM_x, relative_COM_y, 
                    relative_COM_z, old_ligand_COM_x, old_ligand_COM_y, 
                    old_ligand_COM_z, accel_COM_x, accel_COM_y, accel_COM_z,
                    old_ligand_step, receptor_step, accel_step)
                
        if n_frames > 0:
            accel_step = old_ligand_step
            #print("n_frames:", n_frames, "relative_COM_x:", relative_COM_x,
            #      "relative_COM_y:", relative_COM_y, "relative_COM_z:", 
            #      relative_COM_z, "old_ligand_COM_x:", old_ligand_COM_x,
            #      "old_ligand_COM_y:", old_ligand_COM_y, "old_ligand_COM_z",
            #      old_ligand_COM_z)
            assert old_ligand_COM_x is not None, \
                f"Error near file: {ramd_log_filename}, line: {i}"
            n_frames = add_new_frame(
                    trajectory, n_frames, relative_COM_x, relative_COM_y, 
                    relative_COM_z, ligand_COM_x, ligand_COM_y, 
                    ligand_COM_z, accel_COM_x, accel_COM_y, accel_COM_z,
                    old_ligand_step, receptor_step, accel_step)
        trajectories.append(trajectory)
        num_simulations += 1
            
    assert forceOutFreq is not None, \
        "The variable forceOutFreq was not assigned in the log file."
    if positionPrintFreq is None:
        positionPrintFreq = forceOutFreq
    assert forceRAMD is not None, \
        "The variable forceRAMD was not assigned in the log file."
    assert timeStep is not None, \
        "The variable timeStep was not assigned in the log file."
    
    return trajectories, forceOutFreq, positionPrintFreq, forceRAMD, timeStep, \
        num_simulations, temperature, max_dist

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
    
    with open(ramd_log_filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            forceRAMD_search = re.match("RAMD: forceRAMD * (\d+\.?\d*[eE]?[+-]?\d*)", line)
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
            
            timeStep_search = re.match("RAMD: timeStep * (\d+\.?\d*[eE]?[+-]?\d*)", line)
            if timeStep_search:
                timeStep_ = float(timeStep_search.group(1))
                if timeStep is not None:
                    assert timeStep == timeStep_, \
                        "The log files may not have conflicting values of "\
                        "timeStep"
                else:
                    timeStep = timeStep_
            
            if include_all_times:
                ramd_force_search = re.match("RAMD FORCE: (\d+) >", line)
                if ramd_force_search:
                    exit_step = int(ramd_force_search.group(1))
            
            exit_event_search = re.match("EXIT: (\d+)  > MAX DISTANCE LIGAND COM - PROTEIN COM REACHED", line)
            if exit_event_search:
                exit_step = int(exit_event_search.group(1))
            
    end_times.append(exit_step)
    assert forceRAMD is not None, \
        "The variable forceRAMD was not assigned in the log file."
    assert timeStep is not None, \
        "The variable timeStep was not assigned in the log file."
    
    return end_times, forceRAMD, timeStep


# TODO: marked for removal as a method
def get_cv_vec_traj_avg(trajectories):
    """
    This method doesn't tend to work well. MARKED FOR REMOVAL
    """
    sum_ligx = 0.0
    sum_ligy = 0.0
    sum_ligz = 0.0
    total_points = 0
    start_accx = None
    for i, trajectory in enumerate(trajectories):
        for j, frame in enumerate(trajectory):
            [ligx, ligy, ligz, accx, accy, accz] = frame
            #lig_length = np.sqrt(ligx**2 + ligy**2 + ligz**2)
            
            sum_ligx += ligx
            sum_ligy += ligy
            sum_ligz += ligz
            total_points += 1
            if start_accx is None:
                start_accx = accx
            else:
                #assert start_accx == accx, \
                #    "Every frame must have the same acceleration vector"
                pass
    
    assert total_points > 0, "No trajectories detected."
    
    avg_ligx = sum_ligx / total_points
    avg_ligy = sum_ligy / total_points
    avg_ligz = sum_ligz / total_points
    avg_lig_length = np.sqrt(avg_ligx**2 + avg_ligy**2 + avg_ligz**2)
    norm_avgx = avg_ligx / avg_lig_length
    norm_avgy = avg_ligy / avg_lig_length
    norm_avgz = avg_ligz / avg_lig_length
    norm_avg_vec = np.array([norm_avgx, norm_avgy, norm_avgz])
    return norm_avg_vec

def get_mean_vec_for_frames_greater_than_boundary(trajectories, boundary):
    """
    Find a mean near the exit of the site (beyond a certain boundary) to
    help ensure that CVs are going in a positive direction.
    """
    sum_vec = np.zeros(3)
    counter = 0
    for i, trajectory in enumerate(trajectories):
        for j, frame in enumerate(trajectory):
            [ligx, ligy, ligz, accx, accy, accz] = frame
            lig_dist_sq = ligx**2 + ligy**2 + ligz**2
            if lig_dist_sq < boundary**2:
                # Skip all frames too far into the site
                continue
            sum_vec[0] = ligx
            sum_vec[1] = ligy
            sum_vec[2] = ligz
            counter += 1
    mean_vec = sum_vec / counter
    return mean_vec

def get_cv_vec_principal_component(trajectories, mean_vec=None):
    """
    
    """
    n = 0
    for trajectory in trajectories:
        n += len(trajectory)
    data_matrix = np.zeros((3,n))
    counter = 0
    # We are not centering the data - we want starting position to be the
    #  origin
    for i, trajectory in enumerate(trajectories):
        for j, frame in enumerate(trajectory):
            [ligx, ligy, ligz, accx, accy, accz] = frame
            data_matrix[0,counter] = ligx
            data_matrix[1,counter] = ligy
            data_matrix[2,counter] = ligz
            counter += 1
    
    if mean_vec is None:
        mean_vec = np.mean(data_matrix, axis=1)
    cov = np.cov(data_matrix)
    eig_val, eig_vec = np.linalg.eig(cov)
    # Sort the e-vals and find e-vec with highest e-val
    indices = np.arange(0, len(eig_val), 1)
    indices = ([x for _, x in sorted(zip(eig_val, indices))])[::-1]
    eig_val = eig_val[indices]
    eig_vec = eig_vec[:,indices]
    norm_avg_vec = np.zeros(3)
    if np.dot(eig_vec[:,0], mean_vec) > 0.0:
        norm_avg_vec[0] = eig_vec[0,0]
        norm_avg_vec[1] = eig_vec[1,0]
        norm_avg_vec[2] = eig_vec[2,0]
    else:
        norm_avg_vec[0] = -eig_vec[0,0]
        norm_avg_vec[1] = -eig_vec[1,0]
        norm_avg_vec[2] = -eig_vec[2,0]
    
    return norm_avg_vec

def condense_trajectories(trajectories, max_dist, onlyZ=False):
    """
    Reduce trajectories to 1D, and compute xi values
    
    The variable CONDENSE_MODE represents how the ligand position and
    acceleration vector should relate to one another. If CONDENSE_MODE is
    "lig_avg", then the average position of the ligand is taken as the CV
    to which the acceleration vector is dotted (for later binning). This mode
    likely should require one to align the receptor every single frame.
    
    In contrast, "lig_vec" indicates that the CV is merely the vector from
    the center of the site pointing to the COM of the ligand every frame,
    and the acceleration vectors will be dotted to the ligand vector for
    every frame. This mode doesn't require one to align the receptor.
    """
    # First, find the average position of all ligand positions, this will
    # define our 1D CV per trajectory
    CONDENSE_MODE = "prin_comp" # Can be "prin_comp", "lig_avg" or "lig_vec"
    # The cylinder settings are to avoid counting trajectory frames that have
    #  exited the site and have found their way out and around the edges of the
    #  receptor, which might contaminate lower milestones.
    MUST_BE_WITHIN_CYLINDER = True
    CYLINDER_MAX_RADIUS = 0.5 * max_dist
    #if len(trajectory) == 0:
    #    continue
    
    if (CONDENSE_MODE == "lig_avg") and not onlyZ:
        norm_avg_vec = get_cv_vec_traj_avg(trajectories)
        norm_avgx = norm_avg_vec[0]
        norm_avgy = norm_avg_vec[1]
        norm_avgz = norm_avg_vec[2]
    elif (CONDENSE_MODE == "prin_comp") and not onlyZ:
        boundary = 0.5 * max_dist
        mean_vec = get_mean_vec_for_frames_greater_than_boundary(
            trajectories, boundary)
        norm_avg_vec = get_cv_vec_principal_component(trajectories, mean_vec)
        norm_avgx = norm_avg_vec[0]
        norm_avgy = norm_avg_vec[1]
        norm_avgz = norm_avg_vec[2]
    
    one_dim_trajs = []
    for i, trajectory in enumerate(trajectories):
        one_dim_traj = []
        for j, frame in enumerate(trajectory):
            [ligx, ligy, ligz, accx, accy, accz] = frame
            # dot product between frame pos and avg_lig
            if onlyZ:
                cv_val = ligz
                xi_val = accz
            else:
                if CONDENSE_MODE in ["lig_avg", "prin_comp"]:
                    cv_val = (ligx*norm_avgx + ligy*norm_avgy + ligz*norm_avgz)
                    #cv_val = np.sqrt(ligx*ligx + ligy*ligy + ligz*ligz)
                    # dot product between frame acc and avg_lig
                    xi_val = (accx*norm_avgx + accy*norm_avgy + accz*norm_avgz)
                    if MUST_BE_WITHIN_CYLINDER:
                        lig_vec = np.array([ligx, ligy, ligz])
                        projection_vec = norm_avg_vec * np.dot(norm_avg_vec, lig_vec)
                        radial_vec = lig_vec - projection_vec
                        if np.linalg.norm(radial_vec) > CYLINDER_MAX_RADIUS:
                            # If the frame goes outside of the encompassing 
                            # cylinder, then simply end this trajectory
                            break
                    
                elif CONDENSE_MODE == "lig_vec":
                    cv_val = np.sqrt(ligx*ligx + ligy*ligy + ligz*ligz)
                    norm_ligx = ligx / cv_val
                    norm_ligy = ligy / cv_val
                    norm_ligz = ligz / cv_val
                    xi_val = (accx*norm_ligx + accy*norm_ligy + accz*norm_ligz)
                else:
                    raise Exception(f"Method not implemented: {CONDENSE_MODE}")
            one_dim_traj.append([cv_val, xi_val])
            
        one_dim_trajs.append(one_dim_traj)
    
    return one_dim_trajs
