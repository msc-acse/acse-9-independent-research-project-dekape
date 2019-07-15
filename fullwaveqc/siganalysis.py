import numpy as np
import datetime
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import warnings


def wavespec(Wavelet, ms=True, fmax=None, plot=False, n=None):
    """
    Returns and plots the frequency spectrum of a source wavelet.
    :param Wavelet: SegyData object outputted from fullwaveqc.tools.load function
    :param ms: (bool) Set to true if sampling rate in Wavelet object is
    :param fmax: (float) Value of the highest frequency expected in the signal
    :param plot: (bool) Will plot the wavelet in time and wave frequencies if set to True
    :param n: (int) length of output axis from performing FFT. If less than signal size, signal will be cropped, if more, it
    will be padded with zeros
    :return:
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
    yf = fft(signal, n=n)
    if n is not None:
        N = n

    # If expected frequency is given, check for aliasing
    if fmax is not None:
        if fs <= 2*fmax:
            warnings.warn("Sampling frequency (%.3f Hz) does not meet Nyquist requirements (> %.3f Hz). \
                          Possibility of aliased signal"%(fs, 2*fmax))

    # Create and shift the frequnecy domain
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), N // 2)
    xf = fftfreq(N//2, d = 2*dt)

    # Normalise and get energy spectrum
    yf = 2 * np.sqrt(yf.real**2 + yf.imag**2) / N
    # yf = -20*np.log10(yf)

    if plot:
        # Create figure
        figure, ax = plt.subplots(2, 1)
        figure.set_size_inches(18.5, 10.5)

        # Plot time domain
        ax[0].plot(time, signal)
        ax[0].grid()
        ax[0].set(title="Wavelet - Time Domain", xlabel="Time(s)", ylabel="Amplitude")

        # Plot frequency domain
        ax[1].plot(xf, yf[0:N // 2], '.-')
        ax[1].grid()
        ax[1].set(title="Wavelet - Frequency Domain", xlabel="Frequency(Hz)", ylabel="Normalised Amplitude")
        if fmax is not None:
            ax[1].set_xlim(0, 1.2*fmax)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

    return yf, xf


def dataspec(SegyData, ms=True, shot=1):
    data = SegyData.data[shot-1]
    nx, nt = np.shape(data)
    dx, dt = SegyData.dt[shot-1], SegyData.dsrc[0]
    if ms:
        dt = dt/1000.

    # Find next integer power of 2 of nx and nt
    nk, nf = int(2 ** np.ceil(np.log(nx) / np.log(2.0))), int(2 ** np.ceil(np.log(nt) / np.log(2.0)))

    # FK transform (2D Fourier) on the data
    fk = np.fft.fft2(data, s=[nk, nf])

    f = np.linspace(0, 1.0 / (2.0 * dt), nf // 2)
    k = np.linspace(0, 1.0 / (2.0 * dx), nk // 2)

    kk, ff = np.meshgrid(k, f)
    plt.contourf(kk, ff, np.rot90((1/(2*nf)) * (1/(2*nk)) * np.abs(fk[0:nk//2, 0:nf//2])))
    plt.colorbar()
    plt.show()

    return f, k, fk


def phasediff():
    return


def modelspec():
    return