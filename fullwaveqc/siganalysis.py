#!/usr/bin/env python
# Deborah Pelacani Cruz
# https://github.com/dekape
import numpy as np
import datetime
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import warnings
import sys
import fullwaveqc.tools as tools


def closest2pow(n):
    """
    Finds the first integer greater or equal to n that is a power of 2

    Parameters
    ----------
    n: float
        any positive number

    Returns
    -------
    n2pow: int
        the first integer greater or equal to n that is an exact power of two

    Raises
    ------
        ValueError if n is less or equal to zero
    """

    if n <= 0:
        raise ValueError("Can only compute the closest power of two of positive numbers")

    n2pow = int(2 ** np.ceil(np.log(n) / np.log(2.)))
    return n2pow


def gausswindow(samples, wstart, wend, dt):
    """
    Create a gaussian function to window a signal. Standard deviation of the gaussian function is equivalent to one
    quarter the window width

    Parameters
    ----------
    samples: int
        Total of points in the function
    wstart: int
        Start time of the window (ms)
    wend: int
        End time of the window (ms)
    dt: float
        Time sampling of the signal(ms)

    Returns
    -------
    w: numpy.array
        1D array of a gaussian window of size (samples, 1)
    """

    # Transform window to index points
    wend, wstart = wend/dt, wstart/dt

    # Compute width and centering of the window
    fac = 2
    std = 0.5 * (wend - wstart) / fac
    mean = wstart + fac*std

    # Create samples array and gaussian function
    x = np.arange(0, samples, 1)
    w = np.exp(-0.5 * ((x - mean) / std) ** 2)

    return w


def wavespec(Wavelet, ms=True, fmax=None, plot=False, fft_smooth=3):
    """
    Returns and plots the frequency spectrum of a source wavelet.

    Parameters
    ----------
    Wavelet: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function with the wavelet data
    ms: bool, optional
        Set to true if sampling rate in Wavelet object is in miliseconds, otherwise assumed
        in seconds. Default: True
    fmax: float, optional
        Value of the highest frequency expected in the signal. Default: None
    plot: bool, optional
        Will plot the wavelet in time and wave frequencies if set to True. Default: False
    fft_smooth: int, optional
        Parameter used to multiply the number of samples inputted into the Fast Fourier
        Transform. Increase this factor for a smoother plot. The final number of sample
        points will be the nearest power of two of fft_smooth multiplied by the original
        number of time samples in the signal. Higher value increases computational time.
        Default: 3

    Returns
    -------
    xf: numpy.array
        1D array containing the frequencies
    yf: numpy.array
        1D array containing the power of the frequencies in dB
    phase: numpy.array
        1D array containing the unwrapped phases at each frequency

    """

    # Obtaining information from the wavelet
    n = Wavelet.samples[0]
    dt = Wavelet.dt[0]
    if ms:
        dt = dt / 1000.
    time = np.linspace(0, n * dt, n)
    signal = Wavelet.data[0][0]

    # Compute sampling frequency
    fs = 1./dt
    sys.stdout.write("\n" + str(datetime.datetime.now()) + " \t Sampling Frequency = %.4f Hz" % fs)

    # Perform fft and create frequency domain
    n = closest2pow(fft_smooth * n)
    yf = fft(signal, n=n)

    # If expected frequency is given, check for aliasing
    if fmax is not None:
        if fs <= 2*fmax:
            warnings.warn("Sampling frequency (%.3f Hz) does not meet Nyquist requirements (> %.3f Hz). \
                          Possibility of aliased signal" % (fs, 2*fmax))

    # Create and shift the frequency domain
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), n // 2)

    # Get phase the normalise amplitudes to get energy spectrum
    phase = np.arctan(yf.imag / yf.real)
    phase = np.unwrap(2 * phase) / 2
    yf = 2 * np.sqrt(yf.real**2 + yf.imag**2) / n

    # crop yf and phase to fit frequency domain
    yf = yf[0:n // 2]
    phase = phase[0:n // 2]

    # normalise frequency spectrum to dB
    yf = 20*np.log10(yf/np.max(yf))

    if plot:
        # Create figure
        figure, ax = plt.subplots(3, 1)
        figure.set_size_inches(11.5, 11.5)

        # Plot time domain
        ax[0].plot(time, signal)
        ax[0].grid()
        ax[0].set(title=Wavelet.name, xlabel="Time(s)", ylabel="Amplitude")
        # ax[0].set_xlim(0, time[np.where(signal > 0)[0][-1]])

        # Plot frequency domain
        if fmax is not None:
            idx = (np.abs(xf - 1.2*fmax)).argmin()
        else:
            idx = -1
        ax[1].plot(xf[:idx], yf[:idx], '.-')
        ax[1].grid()
        ax[1].set(title=Wavelet.name, xlabel="Frequency (Hz)", ylabel="Amplitude(dB)")
        ax[1].set_ylim(-80, 1)

        ax[2].plot(xf[:idx], phase[:idx], '.-')
        ax[2].set(title=Wavelet.name, xlabel="Frequency (Hz)", ylabel="Phase (rad)")
        ax[2].grid()
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

    return xf, yf, phase


def dataspec(SegyData, ms=True, shot=1, fmax=None, fft_smooth=3, plot=False):
    """
    Returns and plots the frequency spectrum of a single shot of a dataset. Does so by stacking the frequencies of each
    individual trace.

    Parameters
    ----------
    SegyData: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function containing the dataset
    ms: bool, optional
        Set to true if sampling rate in Wavelet object is in miliseconds, otherwise assumed
        in seconds. Default: True
    shot: int, optional
        Shot number to compute the frequency spectrum. Default 1
    fmax: float, optional
        Value of the highest frequency expected in the signal. Default: None
    plot: bool, optional
        Will plot the wavelet in time and wave frequencies if set to True. Default: False
    fft_smooth: int, optional
        Parameter used to multiply the number of samples inputted into the Fast Fourier
        Transform. Increase this factor for a smoother plot. The final number of sample
        points will be the nearest power of two of fft_smooth multiplied by the original
        number of time samples in the signal. Higher value increases computational time.
        Default: 3

    Returns
    -------
    xf: numpy.array
        1D array containing the frequencies
    yf: numpy.array
        1D array containing the power of the frequencies in dB
    phase: numpy.array
        1D array containing the unwrapped phases at each frequency
    """

    # Get shot information and adjust dt
    nx, nt = np.shape(SegyData.data[shot-1])
    dt = SegyData.dt[shot - 1]
    if ms:
        dt = dt/1000.
    fs = 1. / dt
    sys.stdout.write(str(datetime.datetime.now()) + " \t Sampling Frequency = %.4f Hz" % fs)

    # whole of f domain points (larger than nt for smoothed plot)
    nt = closest2pow(fft_smooth * nt)

    # Create and shift the frequnecy domain
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), nt // 2)

    # Store space to add yf
    yf = np.zeros((nt,), dtype='complex64')

    # Perform fft on every trace and add amplitude to yf
    for ix in range(0, nx):
        yf += fft(SegyData.data[shot-1][ix], n=nt)

    # Get phase the normalise amplitudes to get energy spectrum
    phase = np.arctan(yf.imag / yf.real)
    phase = np.unwrap(2 * phase) / 2
    yf = 2 * np.sqrt(yf.real**2 + yf.imag**2) / nt

    # normalise frequency spectrum to dB
    yf = 20*np.log10(yf/np.max(yf))

    # cut frequencies to fit shifted domain
    phase = phase[0:nt // 2]
    yf = yf[0:nt // 2]

    if plot:
        # Create figure
        figure, ax = plt.subplots(2, 1)
        figure.set_size_inches(11.5, 8.5)

        # Plot frequency domain
        if fmax is not None:
            idx = (np.abs(xf - 1.2*fmax)).argmin()
        else:
            idx = -1

        ax[0].plot(xf[:idx], yf[:idx], '.-')
        ax[0].grid()
        ax[0].set(title=SegyData.name + "-Shot %g - Frequency Domain" % shot, xlabel="Frequency (Hz)",
                  ylabel="Amplitude (dB)")
        ax[0].set_ylim(-80, 1)

        ax[1].plot(xf[:idx], phase[:idx], '.-')
        ax[1].set(title=SegyData.name + "-Shot %g - Frequency Domain" % shot, xlabel="Frequency (Hz)",
                  ylabel="Phase (rad)")
        ax[1].grid()

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

    return xf, yf, phase


def phasediff(PredData, ObsData, f=1, wstart=200, wend=1000, nr_min=0, nr_max=None, ns_min=0, ns_max=None, ms=True,
              fft_smooth=3, unwrap=True, plot=False, verbose=1):
    """
    Computes and plots the phase difference between an observed and predicted dataset, at a single specified frequency,
    for all receivers and shots. Will present undesired unwrapping effects in the presence of noise or low-amplitude
    signal. Calculates phase_observed - phase_predicted.

    Parameters
    ----------
    PredData: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function
    ObsData: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function. Should have the same time sampling as PredData
    f: float, optional
        Frequency in Hz at which the phase difference should be calculated. Default: 1
    wstart: int, optional
         Time sample to which start the window for the phase difference computation. Default: 200
    wend: int, optional
        Time sample to which end the window for the phase difference computation.If negative will take the entire
        shot window. Default: 1000
    nr_min: int, optional
        Minimum receiver index to which calculate the phase difference. Default 0.
    nr_max: int, optional
        Maximum  receiver index to which calculate the phase difference. If None is given, then number of receivers
         is inferred from the datasets. Default None.
    ns_min: int, optional
        Minimum source/shot index to which calculate the phase difference. Default: 0
    ns_max: int, optional
        Maximum source/shot index to which calculate the phase difference. If None is
        given then number of sources is inferred from the datasets. Default: None
    ms: bool, optional
        Set to true if sampling rate in Wavelet object is in miliseconds, otherwise assumed
        in seconds. Default: True
    fft_smooth: int, optional
        Parameter used to multiply the number of samples inputted into the Fast Fourier
        Transform. Increase this factor for a smoother plot. The final number of sample
        points will be the nearest power of two of fft_smooth multiplied by the original
        number of time samples in the signal. Higher value increases computational time.
        Default: 3
    unwrap: bool, optional
        If set to true will perform phase unwrapping in the receiver domain. For a detailed
        discussion of how these might affect the phase difference results, refer to the
        project report in the Github repository. Default True
    plot: bool, optional
        Will plot the phase difference if set to True. Default: False
    verbose: bool, optional
        If set to True will verbose the main steps of the function calculation. Default True.

    Returns
    -------
    phase_pred: numpy.array
        2D array of size (ns_max, nr_max) with the unwrapped phases of the
        predicted dataset at the specified frequency
    phase_obs: numpy.array
        2D array of size (ns_max, nr_max) with the unwrapped phases of the
        observed dataset at the specified frequency
    phase_diff: numpy.array
        phase_diff  2D array of size (ns_max-ns_min, nr_max-nrmin) with the unwrapped phase differences
        between the observed and predicted datasets at the specified frequency

    """

    # Set verbose
    verbose_print = tools.set_verbose(verbose)

    # Get number of sources and max number of receivers per source
    if ns_max is None:
        ns_max = PredData.nsrc

    if nr_max is None:
        nr_max = np.max(np.array(PredData.nrec))

    # Reserve space to store phases
    phase_pred = np.zeros((ns_max-ns_min, nr_max-nr_min))
    phase_obs = np.zeros((ns_max-ns_min, nr_max-nr_min))

    # Loop through each shot -- try and except
    for i in range(ns_min, ns_max):
        # Get time sampling from data and compute sampling frequency for FFT
        dt = PredData.dt[i]
        if ms:
            dt = dt / 1000.

        if wend <= 0:
            wend = PredData.samples[i] * PredData.dt[i]

        # Get number of samples for FFT -- multiples of 2 are significantly faster
        nt = PredData.samples[i]
        nt = closest2pow(fft_smooth * nt)

        # Create Gaussian window
        w = gausswindow(PredData.samples[i], wstart, wend, PredData.dt[i])

        # Create and shift the frequency domain
        xf = np.linspace(0.0, 1.0 / (2.0 * dt), nt // 2)
        # print(xf)

        # Find array index in frequency domain closest to frequency of interest
        idf = (np.abs(xf - f)).argmin()
        verbose_print(str(datetime.datetime.now()) + " \t Calculating phase at frequency %.2fHz of shot %g...\r" %
                      (xf[idf], i))

        try:
            # Loop through each receiver -- try and except
            for j in range(nr_min, nr_max):
                # compute phase for predicted dataset
                try:
                    # multiply pred and obs trace by gauss window
                    pred_trace = (w * PredData.data[i][j])
                    obs_trace = (w * ObsData.data[i][j])

                    # do fft for pred and obs separately, slice it to match frequency domain xf
                    pred_fft = fft(pred_trace, n=nt)[0:nt // 2]
                    obs_fft = fft(obs_trace, n=nt)[0:nt // 2]

                    # compute and unwrap phase
                    if pred_fft.real[idf] == 0:
                        raise RuntimeError
                    else:
                        phase_p = np.arctan(pred_fft.imag / pred_fft.real)
                        phase_p = np.unwrap(2 * phase_p) / 2

                        # store phase for pred and obs at single frequency
                        phase_pred[i-ns_min, j-nr_min] = phase_p[idf]

                    # compute and unwrap phase
                    if obs_fft.real[idf] == 0:
                        raise RuntimeError
                    else:
                        phase_o = np.arctan(obs_fft.imag / obs_fft.real)
                        phase_o = np.unwrap(2 * phase_o) / 2

                        # store phase for pred and obs at single frequency
                        phase_obs[i-ns_min, j-nr_min] = phase_o[idf]

                except (RuntimeError, IndexError):
                    pass  # value phase differences to zero
        except IndexError:
            pass

    verbose_print(str(datetime.datetime.now()) + "                   \t All phases calculated successfully \n")

    # Unwrap phase in space and Compute phase difference
    if unwrap:
        # unwrap in receiver domain
        phase_obs = np.unwrap(2*phase_obs, axis=1, discont=np.pi)/2
        phase_pred = np.unwrap(2*phase_pred, axis=1, discont=np.pi)/2
    phase_diff = phase_obs - phase_pred

    # Plot
    if plot:
        s = np.arange(0, ns_max - ns_min, 1) + ns_min
        r = np.arange(0, nr_max - nr_min, 1) + nr_min

        cmap = "RdYlGn"
        figure, ax = plt.subplots(1, 1)
        figure.set_size_inches(7.5, 7.5)
        ax.contourf(r, s, phase_diff * 180/np.pi, cmap=cmap, levels=360, vmin=-180., vmax=180.)

        ax.set(xlabel="Rec x", ylabel="Src x")
        ax.set_title(PredData.name + " f=%.2f Hz %g ms - %g ms" % (f, wstart, wend), pad=40)
        ax.invert_yaxis()
        ax.tick_params(labeltop=True, labelright=False, labelbottom=True, labelleft=True)

        # Format limits of colorbar
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(phase_diff)
        m.set_clim(-180., 180.)
        cbar = plt.colorbar(m)
        cbar.set_label(r"$\Delta \phi$ (°)")

        plt.show()
    return phase_pred, phase_obs, phase_diff


def xcorr(PredData, ObsData, wstart=0, wend=-1, nr_min=0, nr_max=None, ns_min=0, ns_max=None, ms=True, plot=False,
          verbose=1):
    """
    Computes and plots the cross-correlation between an observed and predicted dataset using numpy.correlate.
    Traces are normalised to unit length for comparison

    Parameters
    ----------
    PredData: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function
    ObsData: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function
    wstart: int, optional
        Time sample to which start the window for the phase difference computation. Default: 0
    wend: int, optional
        Time sample to which end the window for the phase difference computation.If negative will use the entire shot
        window. Default -1.
    nr_min: int, optional
        Minimum receiver index to which calculate the phase difference. Default 0.
    nr_max: int, optional
        Maximum  receiver index to which calculate the phase difference. If None is given, then number of receivers
         is inferred from the datasets. Default None.
    ns_min: int, optional
        Minimum source/shot index to which calculate the phase difference. Default: 0
    ns_max: int, optional
        Maximum source/shot index to which calculate the phase difference. If None is
        given then number of sources is inferred from the datasets. Default: None
    ms: bool, optional
        Set to true if sampling rate in Wavelet object is in miliseconds, otherwise assumed
        in seconds. Default: True
    plot: bool, optional
        Will plot the phase difference if set to True. Default: False
    verbose: bool, optional
        If set to True will verbose the main steps of the function calculation. Default True.

    Returns
    -------
    xcorr_arr: numpy.array
        2D array of size (ns_max, nr_max) with the normalised cross correlation values

    """

    # Set verbose
    verbose_print = tools.set_verbose(verbose)

    # Get number of sources and max number of receivers per source
    if ns_max is None:
        ns_max = PredData.nsrc
    if nr_max is None:
        nr_max = np.max(np.array(PredData.nrec))

    # Reserve space to store correlation values
    xcorr_arr = np.zeros((ns_max - ns_min, nr_max - nr_min))

    # Loop through each shot -- try and except
    for i in range(ns_min, ns_max):

        if wend <= 0:
            wend = PredData.samples[i] * PredData.dt[i]

        # Create Gaussian window
        w = gausswindow(PredData.samples[i], wstart, wend, PredData.dt[i])

        # Find array index in frequency domain closest to frequency of interest
        verbose_print(str(datetime.datetime.now()) + " \t Cross correlating traces of shot %g ...\r" % i)

        try:
            # Loop through each receiver -- try and except
            for j in range(nr_min, nr_max):
                # compute phase for predicted dataset
                try:
                    # multiply normalised (to unit length) pred and obs trace by gauss window
                    pred_trace = (w * PredData.data[i][j])
                    obs_trace = (w * ObsData.data[i][j])

                    # cross correlate at zero lag and normalise to [-1, 1] range
                    xcorr_arr[i-ns_min][j-nr_min] = np.sum(pred_trace * obs_trace) / \
                                      np.sqrt(np.sum(pred_trace**2)*np.sum(obs_trace**2))

                except RuntimeError:
                    verbose_print("All zero predicted signal encountered at trace %g of shot %g" % (j, i))
                except IndexError:
                    warnings.warn("Predicted trace %g of shot %g not well defined" % (j, i))
        except IndexError:
            warnings.warn("Shot %g not well defined" % i)

    verbose_print(str(datetime.datetime.now()) + "                   \t All cross-correlations calculated "
                  "successfully \n")

    # Plot
    if plot:
        s = np.arange(0, ns_max - ns_min, 1) + ns_min
        r = np.arange(0, nr_max - nr_min, 1) + nr_min

        cmap="PuRd"
        figure, ax = plt.subplots(1, 1)
        figure.set_size_inches(7.5, 7.5)
        ax.contourf(r, s, xcorr_arr, cmap=cmap, levels=360, vmin=0., vmax=1)

        ax.set(xlabel="Rec x", ylabel="Src x")
        ax.set_title(PredData.name + "XCorr %g ms - %g ms" % (wstart, wend), pad=40)
        ax.invert_yaxis()
        ax.tick_params(labeltop=True, labelright=False, labelbottom=True, labelleft=True)

        # Format limits of colour bar
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(xcorr_arr)
        m.set_clim(0., 1)
        cbar = plt.colorbar(m)
        cbar.set_label(r"Zero Lag Cross Correlation")

        plt.show()

    return xcorr_arr
