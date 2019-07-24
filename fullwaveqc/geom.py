import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
from collections import OrderedDict


def boundarycalc(d, dx, fmin, vmax):
    """

    :param d: (float) distance between source and first receiver in units of distance (meters)
    :param dx: (float)  size of model cell in units of distance (meters)
    :param fmin: (float) minimum frequency present in the wavelet (Hz)
    :param vmax: (float) maximum P-wave velocity expected from the models (m/s)
    :return:
    """

    # Largest wavelength (in units of distance)
    lambda_max = vmax / fmin

    # Number of boundaries necessary to cover one Fresnel radius
    ndist = int(np.ceil(0.5 * np.sqrt(lambda_max * d) / dx))
    if ndist < 4:
        ndist = 4

    # Number of absorbing boundaries required to cover 2 of the largest wavelengths
    nabsorb = int(np.ceil(2 * lambda_max / dx))

    return nabsorb, ndist


def surveygeom(rcvgeopath, srcgeopath, src_list=[], runfilepath=None, plot=False, verbose=0):

    # Store source positions
    if verbose:
        sys.stdout.write(str(datetime.datetime.now()) + " \t Reading source locations ...\r")
    srcx, srcy = [], []
    with open(srcgeopath) as srcgeo:
        for i, line in enumerate(srcgeo):
            if i!=0:
                srcx.append(float(line.split()[1]))
    src = np.array(srcx)

    # Store receiver positions
    if verbose:
        sys.stdout.write(str(datetime.datetime.now()) + " \t Reading receiver locations ...\r")
    rcvx, rcvy = [], []
    with open(rcvgeopath) as rcvgeo:
        for i, line in enumerate(rcvgeo):
            if i!=0:
                rcvx.append(float(line.split()[1]))
    rcvx = np.array(rcvx)

    if verbose:
        sys.stdout.write(str(datetime.datetime.now()) + " \t Rearranging receiver by sources ...\r")
    rcvx_2 = []
    # Rearrange receivers list by source
    src_index = [0]
    for i in range(1, np.size(rcvx)):
        if rcvx[i-1] > rcvx[i]:
            src_index.append(i)
    src_index.append(-1)
    for i in range(0, len(src_index)-1):
        rcvx_2.append(np.array(rcvx[src_index[i]: src_index[i+1]]))

    if verbose:
        sys.stdout.write("                                                                            ")

    # If list of sources not given, create list with all sources
    if len(src_list) == 0:
        src_list = [i for i in range(1, len(srcx) + 1)]

    # Plot every source in list
    if plot:
        figure, ax = plt.subplots(1, 1)
        figure.set_size_inches(15.5, 7.5)
        for i in src_list:
            # Dealing with negative numbers in the list
            if i > 0:
                i -=1
            if i < 0:
                i = len(srcx) + i

            ax.scatter(srcx[i], i + 1, c="y", label="Source", marker="*", s=155)
            ax.scatter(rcvx_2[i], np.zeros_like(rcvx_2[i]) + i + 1, c="r", label="Receiver", marker=11, s=40)

        ax.set_xlim(0, )
        ax.set(xlabel="Lateral offset (m)", ylabel="Shot Number")
        ax.grid()

        # Configuring legend so that it doesn't repeat the labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        ax.legend(by_label.values(), by_label.keys(), loc="best")

        plt.show()

    return srcx, rcvx_2
