"""
parser.py

Parse RAMD log files, and use the log files to construct trajectories of
CVs.
"""

import re
import os

import numpy as np

def parse_ramd_log_file(ramd_log_filename):
    assert os.path.exists(ramd_log_filename), \
        f"File not found: {ramd_log_filename}."
    trajectories = []
    trajectory = []
    n_frames = 0
    relative_COM_x = None
    relative_COM_y = None
    relative_COM_z = None
    ligand_step = None
    ligand_COM_x = None
    ligand_COM_y = None
    ligand_COM_z = None
    with open(ramd_log_filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            new_trajectory_frame = False
            new_trajectory = False
            # Determine the log output frequency
            forceOutFreq_search = re.match("RAMD: forceOutFreq * (\d+)", line)
            if forceOutFreq_search:
                forceOutFreq = int(forceOutFreq_search.group(1))
                
            forceRAMD_search = re.match("RAMD: forceRAMD * (\d+\.\d*)", line)
            if forceRAMD_search:
                forceRAMD = float(forceRAMD_search.group(1))
                
            timeStep_search = re.match("RAMD: timeStep * (\d+\.\d*)", line)
            if timeStep_search:
                timeStep = float(timeStep_search.group(1))
            
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
                #accel_COM_x = float(change_accel_search.group(2))
                #accel_COM_y = float(change_accel_search.group(3))
                #accel_COM_z = float(change_accel_search.group(4))
                old_ligand_step = ligand_step
                new_trajectory_frame = True
                new_trajectory = True
            
            if new_trajectory_frame:
                if n_frames > 0:
                    if relative_COM_x is not None:
                        assert old_ligand_step == receptor_step, \
                            "The lines defining receptor and ligand COM are not the same RAMD step."
                        frame = [relative_COM_x, relative_COM_y, relative_COM_z,
                                 accel_COM_x, accel_COM_y, accel_COM_z]
                    else:
                        frame = [old_ligand_COM_x, old_ligand_COM_y, old_ligand_COM_z,
                                 accel_COM_x, accel_COM_y, accel_COM_z]
                        
                    assert old_ligand_step == accel_step, \
                        "The lines defining ligand COM and accel vector are not the same RAMD step."
                    
                    trajectory.append(frame)
                    
                n_frames += 1
                
            if new_trajectory:
                
                trajectories.append(trajectory)
                trajectory = []
                n_frames = 0
            
    return trajectories, forceOutFreq, forceRAMD, timeStep

def condense_trajectories(trajectories):
    """
    Reduce trajectories to 1D, and compute xi values
    """
    # First, find the average position of all ligand positions, this will
    # define our 1D CV
    sum_ligx = 0.0
    sum_ligy = 0.0
    sum_ligz = 0.0
    total_points = 0
    
    for i, trajectory in enumerate(trajectories):
        start_accx = None
        for j, frame in enumerate(trajectory):
            [ligx, ligy, ligz, accx, accy, accz] = frame
            sum_ligx += ligx
            sum_ligy += ligy
            sum_ligz += ligz
            total_points += 1
            if start_accx is None:
                start_accx = accx
            else:
                assert start_accx == start_accx, \
                    "Every frame must have the same acceleration vector"
            
    avg_ligx = sum_ligx / total_points
    avg_ligy = sum_ligy / total_points
    avg_ligz = sum_ligz / total_points
    avg_lig_length = np.sqrt(avg_ligx**2 + avg_ligy**2 + avg_ligz**2)
    norm_avgx = avg_ligx / avg_lig_length
    norm_avgy = avg_ligy / avg_lig_length
    norm_avgz = avg_ligz / avg_lig_length
    
    one_dim_trajs = []
    for i, trajectory in enumerate(trajectories):
        one_dim_traj = []
        for j, frame in enumerate(trajectory):
            [ligx, ligy, ligz, accx, accy, accz] = frame
            # dot product between frame pos and avg_lig
            cv_val = (ligx*norm_avgx + ligy*norm_avgy + ligz*norm_avgz)
            # dot product between frame acc and avg_lig
            xi_val = (accx*norm_avgx + accy*norm_avgy + accz*norm_avgz)
            one_dim_traj.append([cv_val, xi_val])
            
        one_dim_trajs.append(one_dim_traj)
        
    return one_dim_trajs