#!/usr/bin/env python
# Deborah Pelacani Cruz
# https://github.com/dekape


import context
import fullwaveqc.tools as tools
import os
import numpy as np
import copy


def test_thisfunction():
    assert 1


def test_load_segy():
    dir_path = os.path.abspath(os.path.dirname(__file__))

    # Test reading of Synthetics outputted by Fullwave3D
    segy_path = os.path.join(dir_path, "test_data/PARBASE25FOR2-Synthetic.sgy")
    segy = tools.load(segy_path, model=False, verbose=1)
    assert segy.name == "PARBASE25FOR2-Synthetic"
    assert (segy.src_pos == np.array([1000, 1025])).all()
    assert segy.nsrc == 2
    assert segy.nrec == [801, 801]
    assert len(segy.data) == 2
    assert segy.dt == [2., 2.]

    return


def test_load_vpmodel():
    dir_path = os.path.abspath(os.path.dirname(__file__))

    # Test reading of the Vp models outputted by Fullwave3D
    model_path = os.path.join(dir_path, "test_data/PARBASE25_7-CP00130-Vp.sgy")
    model = tools.load(model_path, model=True, verbose=1)
    assert model.name == "PARBASE25_7-CP00130-Vp"
    assert model.dx == 25
    assert model.nx == 661
    assert model.nz == 201
    assert (model.data.shape == (201, 661))

    return


def test_ampnorm():
    # sanity check
    dir_path = os.path.abspath(os.path.dirname(__file__))
    pred_path = os.path.join(dir_path, "test_data/PARBASE25FOR2-Synthetic.sgy")
    Pred = tools.load(pred_path, model=False, verbose=0)
    PredNorm = tools.ampnorm(Pred, Pred, ref_trace=0, verbose=0)
    assert (np.allclose(PredNorm.data, Pred.data))


def test_ampnorm2():
    # test normalisation of peak value of reference trace
    dir_path = os.path.abspath(os.path.dirname(__file__))
    pred_path = os.path.join(dir_path, "test_data/PARBASE25FOR2-Synthetic.sgy")
    Pred = tools.load(pred_path, model=False, verbose=0)

    Obs = copy.deepcopy(Pred)
    for i in range(0, len(Obs.data)):
        Obs.data[i] = Obs.data[i]*15

    ref = 0
    PredNorm = tools.ampnorm(Obs, Pred, ref_trace=ref, verbose=0)
    for i in range(0, len(PredNorm.data)):
        # compare the max value of reference trace of each shot
        assert (np.isclose(np.max(PredNorm.data[i][ref]), np.max(Obs.data[i][ref])))
    return None


def test_ddwi():
    dir_path = os.path.abspath(os.path.dirname(__file__))
    pred_path = os.path.join(dir_path, "test_data/PARBASE25FOR2-Synthetic.sgy")
    Pred = tools.load(pred_path, model=False, verbose=0)
    Diff = tools.ddwi(Pred, Pred, Pred, normalise=True, name="DIFF", mon_filepath=None, save=False,
                      save_path="./", verbose=0)
    for i in range(0, len(Diff.data)):
        assert (np.allclose(Diff.data[i], Pred.data[i]))
    return None

