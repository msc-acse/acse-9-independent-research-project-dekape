import numpy as np
import datetime
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import matplotlib.pyplot as plt
import warnings
import sys


def closest2pow(n):
    """
    Finds the first integer greater or equal to n that is a power of 2
    """
    n2pow = int(2 ** np.ceil(np.log(n) / np.log(2.)))
    return n2pow


def gausswindow(samples, wstart, wend, dt):
    """
    Create a gaussian function to window a signal
    :param samples: (int) Total of points in the function
    :param wstart: (int) Start time of the window (ms)
    :param wend: (int) End time of the window (ms)
    :param dt: (float) Time sampling of the signal(ms)
    :return: (numpy.array) 1D array of a gaussian window of size (samples, 1)
    """
    wend, wstart = wend/dt, wstart/dt
    std = 0.5 * (wend - wstart)
    mean = wstart + std
    x = np.arange(0, samples, 1)
    w = np.exp(-0.5 * ((x - mean) / std) ** 2)
    return w


def wavespec(Wavelet, ms=True, fmax=None, plot=False, fft_smooth=1):
    """
    Returns and plots the frequency spectrum of a source wavelet.
    :param Wavelet: SegyData object outputted from fullwaveqc.tools.load function
    :param ms:         (bool)  Set to true if sampling rate in Wavelet object is in miliseconds, otherwise assumed in seconds.
                               Default: True
    :param fmax:       (float) Value of the highest frequency expected in the signal
                               Default: None
    :param plot:       (bool)  Will plot the wavelet in time and wave frequencies if set to True
                               Default: False
    :param fft_smooth: (int)   Parameter used to multiply the number of samples inputted into the Fast Fourier Transform.
                               Increase this factor for a smoother plot. The final number of sample points will be the
                               nearest power of two of fft_smooth multiplied by the original number of time samples in the
                               signal. Higher value increases computational time.
                               Default: 1
    :return: xf (np.array) 1D array containing the frequencies
             yf (np.array) 1D array containing the power of the frequencies in dB
             phase (np.array) 1D array containing the unwrapped phases at each frequency
    """
    # Obtaining information from the wavelet
    N = Wavelet.samples[0]
    dt = Wavelet.dt[0]
    if ms:
        dt = dt / 1000.
    time = np.linspace(0, N * dt, N)
    signal = Wavelet.data[0][0]

    fs = 1./dt
    print(datetime.datetime.now(), " \t Sampling Frequency = %.4f Hz"%fs)

    # Perform fft and create frequency domain
    N = closest2pow(fft_smooth * N)
    yf = fft(signal, n=N)

    # If expected frequency is given, check for aliasing
    if fmax is not None:
        if fs <= 2*fmax:
            warnings.warn("Sampling frequency (%.3f Hz) does not meet Nyquist requirements (> %.3f Hz). \
                          Possibility of aliased signal"%(fs, 2*fmax))

    # Create and shift the frequnecy domain
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), N // 2)

    # Get phase the normalise amplitudes to get energy spectrum
    phase = np.arctan(yf.imag / yf.real)
    phase = np.unwrap(2 * phase) / 2
    yf = 2 * np.sqrt(yf.real**2 + yf.imag**2) / N

    # crop yf and phase to fit frequency domain
    yf = yf[0:N // 2]
    phase = phase[0:N // 2]


    # normalise frequency spectrum to dB
    yf = 20*np.log10(yf/np.max(yf))

    if plot:
        # Create figure
        figure, ax = plt.subplots(3, 1)
        figure.set_size_inches(11.5, 11.5)

        # Plot time domain
        ax[0].plot(time, signal)
        ax[0].grid()
        ax[0].set(title="Wavelet - Time Domain", xlabel="Time(s)", ylabel="Amplitude")

        # Plot frequency domain
        if fmax is not None:
            idx = (np.abs(xf - 1.2*fmax)).argmin()
        else:
            idx = -1
        ax[1].plot(xf[:idx], yf[:idx], '.-')
        ax[1].grid()
        ax[1].set(title="Wavelet - Frequency Domain", xlabel="Frequency (Hz)", ylabel="Power(dB)")
        ax[1].set_ylim(-80, 1)


        ax[2].plot(xf[:idx], phase[:idx], '.-')
        ax[2].set(title="Wavelet - Frequency Domain", xlabel="Frequency (Hz)", ylabel="Phase (rad)")
        ax[2].grid()
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

    return xf, yf, phase


def dataspec(SegyData, ms=True, shot=1, fmax=None, fft_smooth=1, plot=False):
    """
    Returns and plots the frequency spectrum of a single shot of a dataset. Does so by stacking the frequencies of each
    individual trace.
    :param SegyData: SegyData object outputted from fullwaveqc.tools.load function
    :param ms:         (bool)  Set to true if sampling rate in Wavelet object is in miliseconds, otherwise assumed in seconds.
                               Default: True
    :param shot        (int)   Shot number to compute the frequency spectrum
                               Default: 1
    :param fmax:       (float) Value of the highest frequency expected in the signal
                               Default: None
    :param plot:       (bool)  Will plot the spectrum and phases of the dataset in the frequency if set to True
                               Default: False
    :param fft_smooth: (int)   Parameter used to multiply the number of samples inputted into the Fast Fourier Transform.
                               Increase this factor for a smoother plot. The final number of sample points will be the
                               nearest power of two of fft_smooth multiplied by the original number of time samples in the
                               signal. Higher value increases computational time.
                               Default: 1
    :return: xf (np.array) 1D array containing the frequencies
             yf (np.array) 1D array containing the power of the frequencies in dB
             phase (np.array) 1D array containing the unwrapped phases at each frequency
    """

    # Get shot information and adjust dt
    nx, nt = np.shape(SegyData.data[shot-1])
    dt = SegyData.dt[shot - 1]
    if ms:
        dt = dt/1000.
    fs = 1. / dt
    print(datetime.datetime.now(), " \t Sampling Frequency = %.4f Hz" % fs)

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

    # cut frequencies to fit shiffted domain
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
        ax[0].set(title="Data - Frequency Domain", xlabel="Frequency (Hz)", ylabel="Power (dB)")
        ax[0].set_ylim(-80, 1)

        ax[1].plot(xf[:idx], phase[:idx], '.-')
        ax[1].set(title="Data - Frequency Domain", xlabel="Frequency (Hz)", ylabel="Phase (rad)")
        ax[1].grid()

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

    return xf, yf, phase


def phasediff(PredData, ObsData, f=1, wstart=200, wend=1000, Nr_max=None, Ns_max=None, ms=True, fft_smooth = 1,
              plot=False, verbose=1):
    """
    Computes and plots the phase difference between an observed and predicted dataset, at a single specified frequency,
    for all receivers and shots
    :param PredData:   SegyData object outputted from fullwaveqc.tools.load function
    :param ObsData:    SegyData object outputted from fullwaveqc.tools.load function
    :param f:          (float)    Frequency in Hz at which the phase difference should be calculated
                                   Default: 1
    :param wstart:     (int)      Time sample to which start the window for the phase difference computation
                                  Default: 200
    :param wend:       (int)      Time sample to which end the window for the phase difference computation
                                  Default: 1000
    :param Nr_max:     (int)      Maximum number of receivers to which calculate the phase difference. If None is given,
                                  then number of receivers is inferred from the datasets
                                  Default: None
    :param Ns_max:     (int)      Maximum number of sources/shots to which calculate the phase difference. If None is given,
                                  then number of sources is inferred from the datasets
                                  Default: None
    :param ms:         (bool)     Set to true if sampling rate in Wavelet object is in miliseconds, otherwise assumed in
                                  seconds.
                                  Default: True
    :param fft_smooth: (int)      Parameter used to multiply the number of samples inputted into the Fast Fourier Transform.
                                  Increase this factor for a smoother plot. The final number of sample points will be the
                                  nearest power of two of fft_smooth multiplied by the original number of time samples in the
                                  signal. Higher value increases computational time.
                                  Default: 1
    :param plot:       (bool)     Will plot the phase difference if set to True
                                  Default: False
    :param verbose:    (bool)     If set to True will verbose the main steps of the function calculation
    :return:           (np.array) phase_pred  2D array of size (Ns_max, Nr_max) with the unwrapped phases of the
                                  predicted dataset at the specified frequency
                       (np.array) phase_obs  2D array of size (Ns_max, Nr_max) with the unwrapped phases of the
                                  observed dataset at the specified frequency
                       (np.array) phase_diff  2D array of size (Ns_max, Nr_max) with the unwrapped phase differences
                                  between the observed and predicted datasets at the specified frequency

    """
    # "For more accurate frequency increase fft_smooth or "
    # "decrease the window size" )

    # Get number of sources and max number of receivers per source
    if Ns_max is None:
        Ns = PredData.nsrc
    else:
        Ns = Ns_max
    if Nr_max is None:
        Nr_max = np.max(np.array(PredData.nrec))
    else:
        Nr_max = Nr_max

    # Reserve space to store phases
    phase_pred = np.zeros((Ns, Nr_max))
    phase_obs = np.zeros((Ns, Nr_max))


    # Loop through each shot -- try and except
    for i in range(0, Ns):

        # Get time sampling from data and compute sampling frequency for FFT
        dt = PredData.dt[i]
        if ms:
            dt = dt / 1000.
        fs = 1. / dt

        # Get number of samples for FFT -- multiples of 2 are significantly faster
        Nt = PredData.samples[i]
        Nt = closest2pow(fft_smooth * Nt)

        # Create Gaussian window
        w = gausswindow(PredData.samples[i], wstart, wend, dt)

        # Create and shift the frequency domain
        xf = np.linspace(0.0, 1.0 / (2.0 * dt), Nt // 2)

        # Find array index in frequency domain closest to frequency of interest
        idf = (np.abs(xf - f)).argmin()
        if verbose:
            sys.stdout.write(str(datetime.datetime.now()) + " \t Calculating phase at frequency %.2fHz of shot %g...\r"%(xf[idf], i))

        try:
            # Loop through each receiver -- try and except
            for j in range(0, Nr_max):
                # compute phase for predicted dataset
                try:
                    # multiply pred and obs trace by gauss window
                    pred_trace = (w * PredData.data[i][j])

                    # do fft for pred and obs separately, slice it to match frequency domain xf
                    pred_fft = fft(pred_trace, n=Nt)[0:Nt // 2]

                    # compute and unwrap phase
                    if pred_fft.real[idf] == 0:
                        raise RuntimeError
                    else:
                        phase_p = np.arctan(pred_fft.imag / pred_fft.real)
                        phase_p = np.unwrap(2 * phase_p) / 2

                        # store phase for pred and obs at single frequency
                        phase_pred[i, j] = phase_p[idf]

                except RuntimeError:
                    if verbose:
                        print("All zero predicted signal encountered at trace %g of shot %g" % (j, i))
                    phase_pred[i, j] = 0.
                except IndexError:
                    warnings.warn("Predicted trace %g of shot %g not well defined" % (j, i))
                try:
                    # multiply trace by gauss window
                    obs_trace = (w * ObsData.data[i][j])

                    # do fft  slice it to match frequency domain xf
                    obs_fft = fft(obs_trace, n=Nt)[0:Nt // 2]

                    # compute and unwrap phase
                    if obs_fft.real[idf] == 0:
                        raise RuntimeError
                    else:
                        phase_o = np.arctan(obs_fft.imag / obs_fft.real)
                        phase_o = np.unwrap(2 * phase_o) / 2
                        # store phase for pred and obs at single frequency
                        phase_obs[i, j] = phase_o[idf]
                except RuntimeError:
                    if verbose:
                        print("All zero observed signal encountered at trace %g of shot %g" % (j, i))
                    phase_obs[i, j] = 0.
                except IndexError:
                    warnings.warn("Observed trace %g of shot %g not well defined" % (j, i))
        except IndexError:
            warnings.warn("Shot %g not well defined"%i)

    if verbose:
        print(str(datetime.datetime.now()) + "                   \t All phases calculated successfully")

    # Unwrap phase in space and Compute phase difference
    phase_obs = np.unwrap(2*phase_obs, axis=1)/2
    phase_pred = np.unwrap(2*phase_pred, axis=1)/2
    phase_diff = (phase_obs - phase_pred)

    # Plot
    if plot:
        # Get unique receiver positions
        rec_pos_all = []
        for x in PredData.rec_pos:
            rec_pos_all.append(list(x))
        rec_pos_all = np.unique(np.array(rec_pos_all[0]))

        cmap = "RdYlGn"
        figure, ax = plt.subplots(1, 1)
        figure.set_size_inches(7.5, 7.5)
        MP = ax.contourf(phase_diff, cmap=cmap, levels=360, vmin=-1.6, vmax=1.6)
        #rec_pos_all, np.array(PredData.src_pos),
        ax.set(xlabel="Rec x", ylabel="Src x")
        ax.set_title("phase_difference f=%.2f Hz %g ms - %g ms"%(f, wstart, wend), pad=40)
        ax.invert_yaxis()
        ax.tick_params(labeltop=True, labelright=False, labelbottom=True, labelleft=True)

        # Format limits of colorbar
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(phase_diff)
        m.set_clim(-1.6, 1.6)
        cbar = plt.colorbar(m)
        cbar.set_label("Unwrapped Phase (rad)")

        plt.show()
    return phase_pred, phase_obs, phase_diff


def bandpass(trace, flow, fhigh, forder, dt):
    # set up parameters for bandpass filtering
    nyq = 0.5 / dt # nyquist
    low = flow / nyq
    high = fhigh / nyq
    b, a = butter(forder, [low, high], btype='band')

    w, h = freqz(b, a, worN=2000)

    if max(abs(h)>1.01):
        print('!!!  Filter has values > 1. Will cause problems !!!')
        print('Try reducing hte filter order')

    # determine if we are dealing with a trace or gather
    traceShape = np.shape(trace)
    nt = traceShape[0]
    if len(traceShape) == 2:
        nz = traceShape[1]
        filtered = np.zeros(shape=[nt,nz])
        for i in range(nz):
            filtered[:,i] = lfilter(b, a, trace[:,i])
    else: # only 1 trace
        filtered = lfilter(b, a, trace)

    return filtered


def xcorr(SegyData):
    return
