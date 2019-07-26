#!/usr/bin/env python

import fullwaveqc.visual as visual
import fullwaveqc.tools as tools
import numpy as np
import os

def test_thisfunction():
    assert 1


def test_vpwell():
    # Load predicted model and true model for well profile testing
    dir_path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(dir_path, "test_data/PARBASE25_7-CP00130-Vp.sgy")
    model = tools.load(model_path, model=True, verbose=0)
    true_path = os.path.join(dir_path, "test_data/TRUE_enl_Vp_bl_y6250.sgy")
    true_model = tools.load(true_path, model=True, verbose=0)
    true_model.dx = 2.  # fix parameters that load couldn't read from model NOT generated by Fullwave

    # Load values and check it matches
    well = visual.vpwell(model, pos_x=[10000], TrueModel=true_model, plot=False)
    well_path = os.path.join(dir_path, "test_data/well_test.npy")
    well_true = np.load(well_path, allow_pickle=True)

    # Compare predicted and true wells, and rms error values
    for i in range(0, len(well_true)):
        assert(np.allclose(well[i][0], well_true[i][0]))
    return None
