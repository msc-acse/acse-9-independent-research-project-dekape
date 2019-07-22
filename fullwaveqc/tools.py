import numpy as np
import segyio
from shutil import copyfile
import datetime
import copy


class SegyData:
    def __init__(self, filepath, name=None):
        if name is not None:
            self.name = name
        else:
            self.name = filepath.split("/")[-1].split('.')[0]
        self.filepath = filepath
        self.src_pos = None
        self.rec_pos = None
        self.nsrc    = -1
        self.nrec    = -1
        self.data    = None
        self.src_z   = None
        self.rec_z   = None
        self.samples = None
        self.dt      = None
        self.dsrc    = None
        self.drec    = None

    def attr(self):
        v = vars(self)
        return v


class Model:
    def __init__(self, filepath, name=None):
        if name is not None:
            self.name = name
        else:
            self.name = filepath.split("/")[-1].split('.')[0]
        self.filepath = filepath
        self.dx = -1
        self.nx = -1
        self.nz = -1
        self.data = -1
        self.minvp = -1
        self.maxvp = -1

        return


    def attr(self):
        v = vars(self)
        return v


def load(filepath, model, scale=1, verbose=1, resample=0):
    """
        2D surveys, same sampling interval and number of samples for all shots
        Models with square grids
    :param filepath:
    :param mode:
    :param scale:
    :param verbose:
    :param resample:
    :return:
    """

    if not model:
        if resample > 0:
            resample = int(resample)
            dstpath = "".join(filepath.split('.')[:-1]) + "-resamp{}".format(resample) + "ms.sgy"
            if verbose:
                print(datetime.datetime.now(), " \t Creating resampled copy at %s..." % dstpath)
            copyfile(filepath, dstpath)
            filepath = dstpath

        # Create Data object
        segy = SegyData(filepath)

        # Open datafile with segyio
        if verbose:
            print(datetime.datetime.now(), " \t Opening file %s..." % filepath)
        with segyio.open(filepath, mode='r+', ignore_geometry=True) as segyfile:
            if resample > 0:
                if verbose:
                    print(datetime.datetime.now(), " \t Resampling to %g ms..." % resample)
                segyio.tools.resample(segyfile, resample)

            if verbose:
                print(datetime.datetime.now(), " \t Loading data...")
            data = np.asarray([trace * 1.0 for trace in segyfile.trace]) * scale
            src_all = segyfile.attributes(segyio.TraceField.SourceX)[:]
            rec_all = segyfile.attributes(segyio.TraceField.GroupX)[:]
            src_z_all, rec_z_all, samples_all, dt_all = [], [], [], []
            for i in range(0, data.shape[0]):
                src_z_all.append(segyfile.header[i][49])
                rec_z_all.append(segyfile.header[i][41])
                samples_all.append(segyfile.header[i][117])
                dt_all.append(segyfile.header[i][115])


        # Get source numbers and positions
        segy.src_pos = np.unique(src_all)                    # position of each source w.r.t to origin
        segy.nsrc = len(segy.src_pos)                        # number of sources/shots
        if verbose:
            print(datetime.datetime.now(), " \t %g shots found. Loading headers..." % segy.nsrc)

        # Collecting and splitting receiver locations and data
        # segy.rec_pos and segy.data should be a list of arrays, each array corresponding to a source
        # segy.nrec is a list containing the number of receivers (or samples) per shot
        segy.rec_pos, segy.nrec, segy.rec_z, segy.drec = [], [], [], []
        segy.src_z, segy.data = [], []
        segy.samples, segy.dt = [], []

        # Loop for every source position
        for i in range(0, len(segy.src_pos)):
            index = np.where(src_all == segy.src_pos[i])

            segy.rec_pos.append(rec_all[index])
            segy.drec.append(np.diff(segy.rec_pos[-1]))
            segy.nrec.append(len(segy.rec_pos[-1]))
            segy.data.append(data[index, :][0])

            segy.rec_z.append(np.array(rec_z_all)[index])
            segy.src_z.append(np.array(src_z_all)[index][0])
            segy.samples.append(np.array(samples_all)[index][0])
            segy.dt.append(np.array(dt_all)[index][0]/1000.)


        if verbose:
            print(datetime.datetime.now(), " \t %g total traces read" % np.sum(np.array(segy.nrec)))

        # Source spacing
        segy.dsrc = np.diff(segy.src_pos)

        if verbose:
            print(datetime.datetime.now(), " \t Data loaded successfully")

        return segy

    else:
        # Create model object
        model = Model(filepath)

        # Open file with segyio
        if verbose:
            print(datetime.datetime.now(), " \t Opening file %s..." % filepath)
        with segyio.open(filepath, mode='r', ignore_geometry=True) as segyfile:
            if verbose:
                print(datetime.datetime.now(), " \t Reading model...")
            data = np.asarray([trace * 1.0 for trace in segyfile.trace]) * scale
            data = data.T
            src_all = segyfile.attributes(segyio.TraceField.SourceX)[:]

        model.data = np.rot90(data.T)
        model.nx = data.shape[1]
        model.nz = data.shape[0]
        model.dx = src_all[1] - src_all[0]
        if verbose:
            print(datetime.datetime.now(), " \t Successfully loaded model of size %g x %g!"%(model.nz, model.nx))
        model.maxvp = np.max(data)
        model.minvp = np.min(data)

        return model


def ampnorm(SegyData, values=[None, None], update=False, shotiter=True):

    data = copy.deepcopy(SegyData.data)

    if not shotiter:
        if (not values[0]) and (not values[1]):
            data = (data - np.mean(data)) / np.std(data)
        else:
            data = (data - values[0])/ np.std(values[1])

    else:
        for i in range(0, len(data)):
            if (not values[0]) and (not values[1]):
                data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])
            else:
                data[i] = (data[i] - values[0]) / np.std(values[1])

    if update:
        SegyData.data = data

    return data


def ddwi(SegyData):
    return


def smooth():
    return