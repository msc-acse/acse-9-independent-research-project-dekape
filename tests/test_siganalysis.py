import context
import fullwaveqc.siganalysis as sig
import os
import fullwaveqc.tools as tools
import copy
import numpy as np

def test_thisfunction():
    assert (True)


def test_phasediff():
    dir_path = os.path.abspath(os.path.dirname(__file__))

    # Load a set of synthetics
    segy_path = os.path.join(dir_path, "test_data/PARBASE25FOR2-Synthetic.sgy")
    segy = tools.load(segy_path, model=False, verbose=1)

    # Make a copy and double the amplitude of the data
    segy2 = copy.deepcopy(segy)
    for i, d in enumerate(segy.data):
        segy2.data[i] = 2 * d

    # Compute the phase difference between the original and the copy
    phasediff = sig.phasediff(segy, segy2, f=3., wstart=200, wend=1000, fft_smooth=1, plot=False)[2]

    # Assert that the phase different is zero for all shots and all traces
    assert ((phasediff == np.zeros([2, 801])).all())
    return
