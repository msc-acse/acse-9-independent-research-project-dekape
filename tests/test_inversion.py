import context
import fullwaveqc.inversion as inv
import numpy as np
import os


def test_thisfunction():
    assert 1


def test_functional():
    dir_path = os.path.abspath(os.path.dirname(__file__))
    job_path = os.path.join(dir_path, "test_data/PARBASE25_8-job001.log")
    iter, func = inv.functional(job_path, plot=False)
    assert (iter == np.array([1, 2, 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13])).all()
    assert (func == np.array([38270., 36650., 36260., 36500., 36100., 36470., 36290., 36520., 36060., 35990., 33260.,
                             33420., 33280.])).all()
    return


def test_steplen():
    dir_path = os.path.abspath(os.path.dirname(__file__))
    job_path = os.path.join(dir_path, "test_data/PARBASE25_8-job001.log")
    iter, slen = inv.functional(job_path, plot=False)
    # assert (iter == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])).all()
    return