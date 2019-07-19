import context
import fullwaveqc.tools as tools
import os
import numpy as np


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