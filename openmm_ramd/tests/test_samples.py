"""
test_samples.py

"""
import numpy as np

import openmm_ramd.analyze.modules.samples as samples

def test_combine_dest_states():
    Q = np.array([[-1.1, 1.1, 0.0, 0.0],
                  [1.2, -2.5, 1.3, 0.0],
                  [0.0, 1.4, -2.9, 1.5],
                  [0.0, 0.0, 1.6, -1.6]])
    dest_indices = [3]
    old_src_indices = [1,2]
    newQ, new_src_indices, dest_index = samples.combine_dest_states(
        Q, old_src_indices, dest_indices)
    assert dest_index == 3
    assert np.isclose(Q, newQ).all()
    assert new_src_indices == [1,2]
    
    dest_indices2 = [0, 3]
    newQ2, new_src_indices2, dest_index2 = samples.combine_dest_states(
        Q, old_src_indices, dest_indices2)
    assert dest_index2 == 2
    trueQ = np.array([[-2.5, 1.3, 1.2],
                      [1.4, -2.9, 1.5],
                      [1.1, 1.6, -2.7]])
    assert np.isclose(trueQ, newQ2).all()
    assert new_src_indices2 == [0,1]
    return