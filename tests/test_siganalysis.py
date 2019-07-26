#!/usr/bin/env python

import context
import fullwaveqc.siganalysis as sig
import os
import fullwaveqc.tools as tools
import copy
import numpy as np


def test_thisfunction():
    assert (True)


def test_phasediff():
    """
    Sanity test for the phase difference function
    """
    dir_path = os.path.abspath(os.path.dirname(__file__))

    # Load a set of synthetics
    segy_path = os.path.join(dir_path, "test_data/PARBASE25FOR2-Synthetic.sgy")
    segy = tools.load(segy_path, model=False, verbose=1)

    # Make a copy and double the amplitude of the data
    segy2 = copy.deepcopy(segy)
    for i, d in enumerate(segy.data):
        segy2.data[i] = 2 * d

    # Compute the phase difference between the original and the copy
    phasediff = sig.phasediff(segy, segy2, f=3., wstart=200, wend=1000, fft_smooth=1, plot=False, verbose=False)[2]

    # Assert that the phase different is zero for all shots and all traces
    assert ((phasediff == np.zeros([2, 801])).all())
    return


def test_xcorr():
    """
    Sanity test for the xcorr difference function
    """
    dir_path = os.path.abspath(os.path.dirname(__file__))

    # Load a set of synthetics
    segy_path = os.path.join(dir_path, "test_data/PARBASE25FOR2-Synthetic.sgy")
    segy = tools.load(segy_path, model=False, verbose=1)

    # Make a copy and double the amplitude of the data
    segy2 = copy.deepcopy(segy)
    for i, d in enumerate(segy.data):
        segy2.data[i] = 2 * d

    # Compute the phase difference between the original and the copy
    xcorr = sig.xcorr(segy, segy2, wstart=200, wend=1000, plot=False, verbose=False)

    # Assert that the phase different is zero for all shots and all traces
    assert np.allclose(xcorr, np.full([2, 801], 1.))
    return


def test_wavespec():
    dir_path = os.path.abspath(os.path.dirname(__file__))

    # Load wavelet and perform signal analysis
    wavelet_path = os.path.join(dir_path, "test_data/PARBASE25FOR2-RawSign.sgy")
    wavelet = tools.load(wavelet_path, model=False)
    wavelet.dt = [1.]     # fix sampling rate and number of samples
    wavelet.samples = [401]
    wavelet_test = sig.wavespec(wavelet, ms=True, fft_smooth=5, fmax=15, plot=False)
    # np.save(os.path.join(dir_path, "test_data/wavelet_test.npy"), wavelet_test, allow_pickle=True)

    # Load true wavelet spectra
    wavelet_path = os.path.join(dir_path, "test_data/wavelet_test.npy")
    wavelet_true = np.load(wavelet_path, allow_pickle=True)

    for i in range(0, 1):
        assert(np.allclose(wavelet_test[i], wavelet_true[i].astype("float32")))

    return


def test_dataspec():
    dir_path = os.path.abspath(os.path.dirname(__file__))
    OBS_PATH = os.path.join(dir_path, "test_data/ucalc_shot_1.sgy")
    OBS = tools.load(OBS_PATH, model=False, verbose=1)
    OBS.dt = [4]       # fix sampling rate and number of samples
    OBS.samples = [1501]
    dataspec1 = sig.dataspec(OBS, ms=True, fft_smooth=5, fmax=15, plot=False)

    # load true spec
    data_path = os.path.join(dir_path, "test_data/dataspec_test.npy")
    dataspec_true = np.load(data_path, allow_pickle=True)

    assert((dataspec1[0] == dataspec_true[0]).all())

    return None


def test_gausswindow():
    w = sig.gausswindow(samples=11, wstart=2, wend=5, dt=2)
    arr = np.array([6.57285286e-02, 6.06530660e-01, 9.45959469e-01, 2.49352209e-01,
           1.11089965e-02, 8.36483472e-05, 1.06453714e-07, 2.28973485e-11,
           8.32396968e-16, 5.11442373e-21, 5.31109225e-27])
    assert(np.allclose(w, arr))
    return None



