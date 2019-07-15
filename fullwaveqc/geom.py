import numpy as np
import matplotlib.pyplot as plt


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


def surveygeom(rcvgeopath, srcgeopath, runfilepath=None):

    # Store source positions
    srcx, srcy = [], []
    with open(srcgeopath) as srcgeo:
        for i, line in enumerate(srcgeo):
            if i!=0:
                srcx.append(float(line.split()[1]))
    src = np.array(srcx)

    # Store receiver positions
    rcvx, rcvy = [], []
    with open(rcvgeopath) as rcvgeo:
        for i, line in enumerate(rcvgeo):
            if i!=0:
                rcvx.append(float(line.split()[1]))
    rcvx = np.array(rcvx)

    rcvx_2 = []
    # Rearrange receivers list by source
    src_index = [0]
    for i in range(1, np.size(rcvx)):
        if rcvx[i-1] > rcvx[i]:
            src_index.append(i)
    src_index.append(-1)

    for i in range(0, len(src_index)-1):
        rcvx_2.append(np.array(rcvx[src_index[i]: src_index[i+1]]))

    for i in range(0, len(srcx)):
        plt.scatter(srcx[i], i*10, c="y")
        plt.scatter(rcvx_2[i], np.zeros_like(rcvx_2[i]) + i*10, c="r")
    plt.show()
    return
