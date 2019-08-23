#!/usr/bin/env python
# Deborah Pelacani Cruz
# https://github.com/dekape
import numpy as np
import segyio
import datetime
import copy
import warnings
import sys
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import os


class SegyData:
    """
    Class to store properties of SEG-Y data files. Should be called only by the fullwaveqc.tools.load function.
    """
    def __init__(self, filepath, name=None):
        if name is not None:
            self.name = name
        else:
            self.name = filepath.split("/")[-1].split('.')[0]
        self._type = "data"
        self._filepath = filepath
        self.src_pos = None
        self.rec_pos = None
        self.nsrc = None
        self.nrec = None
        self.src_z = None
        self.rec_z = None
        self.samples = None
        self.dt = None
        self.dsrc = None
        self.drec = None
        self.data = None

    def attr(self, attr_name=None):
        v = vars(self)
        if attr_name is not None:
            try:
                v = v[attr_name]
            except KeyError:
                v = None
                warnings.warn("Attribute \"%s\" not found. Returning None" % attr_name)
        return v


class Model:
    """
    Class to store properties of SEG-Y 2D model files. Should be called only by the fullwaveqc.tools.load function.
    """
    def __init__(self, filepath, name=None):
        if name is not None:
            self.name = name
        else:
            self.name = filepath.split("/")[-1].split('.')[0]
        self._type = "model"
        self._filepath = filepath
        self.dx = None
        self.nx = None
        self.nz = None
        self.minvp = None
        self.maxvp = None
        self.data = None

        return

    def attr(self, attr_name=None):
        v = vars(self)
        if attr_name is not None:
            try:
                v = v[attr_name]
            except KeyError:
                v = None
                warnings.warn("Attribute \"%s\" not found. Returning None" % attr_name)
        return v


def set_verbose(verbose):
    """
    Function to set the verbose print level
    Parameters
    ----------
    verbose: bool
        Verbose level, set to true to display verbose

    Returns
    -------
    verbose_print(message): lamda function
        Function to display verbose message in console

    """
    if verbose:
        verbose_print = lambda message: sys.stdout.write(message)
    else:
        verbose_print = lambda message: None
    return verbose_print


def load(filepath, model, name=None, scale=1, verbose=1):
    """
    Loads a SEG-Y file into a SegyData or Model class, used for the other functionalities of the fullwaveqc package.
    Accepts 2D segy files and of the same sampling interval for all shots and segy model files with square cells.
    It is HIGHLY recommended that the attributes of this function are checked manually after loading, as different
    SEG-Y header formats are likely to be loaded incorrectly. This function is adapted to the SEGY header format of
    the outputs of Fullwave3D.

    Parameters
    ----------
    filepath: str
        Path to segy file. Must end in .sgy
    model: bool
        Set true to load a Model and false to load a SegyData object
    name: str, optional
        Name to give the data/model object. If None is given, name is inferred from .sgy file. Default is None
    scale: int, optional
        Value to multiply the data content of the files. Default 1
    verbose: bool, optional
        Set to true in order to verbose the steps of this loading function. Default True
    Returns
    -------
    segy or model: fullwaveqc.tools.SegyData or fullwaveqc.tools.Model, depending on parameter model given
    """

    # Set verbose level
    verbose_print = set_verbose(verbose)

    if not model:
        # Create Data object
        segy = SegyData(filepath, name)

        # Open datafile with segyio
        verbose_print("\n" + str(datetime.datetime.now()) + " \t Opening file %s..." % filepath)
        with segyio.open(filepath, mode='r', ignore_geometry=True) as segyfile:
            verbose_print("\n" + str(datetime.datetime.now()) + " \t Loading data...")
            data = np.asarray([trace * 1.0 for trace in segyfile.trace]) * scale
            src_all = segyfile.attributes(segyio.TraceField.SourceX)[:]
            rec_all = segyfile.attributes(segyio.TraceField.GroupX)[:]
            src_z_all, rec_z_all, samples_all, dt_all = [], [], [], []
            for i in range(0, data.shape[0]):
                src_z_all.append(segyfile.header[i][49])
                rec_z_all.append(segyfile.header[i][41])
                samples_all.append(segyfile.header[i][117])
                dt_all.append(segyfile.header[i][115])
                if dt_all[-1] <= 0:
                    warnings.warn("Zero or negative sampling rate. File might not have been loaded correctly."
                                  "Check manually!")

        # Get source numbers and positions
        segy.src_pos = np.unique(src_all)                    # position of each source w.r.t to origin
        segy.nsrc = len(segy.src_pos)                        # number of sources/shots
        verbose_print("\n" + str(datetime.datetime.now()) + " \t %g shots found. Loading headers..." % segy.nsrc)

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

        verbose_print("\n" + str(datetime.datetime.now()) + " \t %g total traces read" %
                      np.sum(np.array(segy.nrec)))

        # Source spacing
        segy.dsrc = np.diff(segy.src_pos)

        verbose_print("\n" + str(datetime.datetime.now()) + " \t Data loaded successfully")

        return segy

    else:
        print("HELLOOOO")
        # Create model object
        model = Model(filepath, name)

        # Open file with segyio
        verbose_print("\n" + str(datetime.datetime.now()) + " \t Opening file %s..." % filepath)
        with segyio.open(filepath, mode='r', ignore_geometry=True) as segyfile:
            verbose_print("\n" + str(datetime.datetime.now()) + " \t Reading model...")
            data = np.asarray([trace * 1.0 for trace in segyfile.trace]) * scale
            data = data.T
            src_all = segyfile.attributes(segyio.TraceField.SourceX)[:]

        model.data = np.rot90(data.T)
        model.nx = data.shape[1]
        model.nz = data.shape[0]
        model.dx = src_all[1] - src_all[0]
        if model.dx <= 0:
            warnings.warn("Zero or negative model spacing. File might not have been loaded correctly."
                          "Check manually!")
        verbose_print("\n" + str(datetime.datetime.now()) + " \t Successfully loaded model of size %g x %g!"
                      % (model.nz, model.nx))
        model.maxvp = np.max(data)
        model.minvp = np.min(data)

        return model


def rm_empty_traces(filename, scale=1, verbose=1):
    """
    Removes all-zero traces of a SEG-Y file and saves the copy of the clean file in the same directory

    Parameters
    ----------
    filename: str
        Path to SEG-Y file, must end in .sgy
    scale: int, optional
        Value to multiply the data content of the files. Default 1
    verbose: bool, optional
        Set to true in order to verbose the steps of this loading function. Default True
    Returns
    -------
    None
    """

    # Set verbose level
    verbose_print = set_verbose(verbose)

    # Destination file name
    dstpath = filename[:-4]+"-CLEAN.sgy"

    verbose_print("\n" + str(datetime.datetime.now()) + " \t Opening file %s..." % filename)
    with segyio.open(filename, mode='r', ignore_geometry=True) as segyfile:
        verbose_print("\n" + str(datetime.datetime.now()) + " \t Reading trace data...")
        data = np.asarray([trace * 1.0 for trace in segyfile.trace])*scale  # (number of traces, number of samples)

        verbose_print("\n" + str(datetime.datetime.now()) + " \t Reading headers...")
        header = []
        for i in range(0, data.shape[0]):
            header.append(segyfile.header[i])  # list of dictionaries

        verbose_print("\n" + str(datetime.datetime.now()) + " \t Collecting header parameters...")
        newspec = segyio.spec()
        newspec.ilines = segyfile.ilines
        newspec.xlines = segyfile.ilines
        newspec.samples = segyfile.samples
        newspec.sorting = segyfile.sorting
        newspec.format = segyfile.format

    verbose_print("\n" + str(datetime.datetime.now()) + " \t Cleaning traces and headers ...")
    newdata = data[~np.all(data == 0, axis=1)]
    newheader = []
    for i in range(0, data.shape[0]):
        if not (data[i, :] == 0).all():
            newheader.append(header[i])
    assert newdata.shape[0] == len(newheader)

    verbose_print("\n" + str(datetime.datetime.now()) + " \t Removed %g empty traces!" %
                  (data.shape[0] - newdata.shape[0]))
    newspec.tracecount = newdata.shape[0]

    verbose_print("\n" + str(datetime.datetime.now()) + " \t Writing trace and headers to new file at %s..." % dstpath)
    with segyio.create(dstpath, newspec) as newsegy:
        newsegy.trace[:] = newdata
        for i in range(0, newdata.shape[0]):
            newsegy.header[i] = newheader[i]

    verbose_print("\n" + str(datetime.datetime.now()) + " \t Clean file created successfully!")

    return None


def ddwi(MonObs, BaseObs, BasePred, normalise=True, name=None, mon_filepath=None, save=False, save_path="./",
         verbose=1):
    """
    Creates the monitor dataset used as the observed dataset for the Double Difference Waveform Inversion.
    The DDWI monitor dataset is given by: u_mon = d_mon - d_base + u_base
    Where d_mon and d_base are the observed monitor and baseline datasets, respectively, and u_base is the dataset from
    a predicted baseline inversion model.
    Note: If data delay has been applied to BaseObs and MonObs, the resulting ddwi dataset will already have the data
    delay applied.

    Parameters
    ----------
    MonObs: fullwaveqc.tools.SegyData
        Object outputted from fullwaveqc.tools.load function for the true monitor data
    BaseObs: fullwaveqc.tools.SegyData
        Object outputted from fullwaveqc.tools.load function for the true baseline data.
        Requires same number of samples, sampling interval, shots and receiver position
        as MonObs
    BasePred: fullwaveqc.tools.SegyData
        Object outputted from fullwaveqc.tools.load function for the predicted baseline data.
        Requires same number of samples, sampling interval, shots and receiver position
        as MonObs
    normalise: bool, optional
        If true will normalise the amplitude of the predicted baseline dataset to match
        the first trace of the observed dataset. Does so shot by shot. Default True
    name: str, optional
        Name to give the DDWI dataset, if None will give the name "DDWI-DATASET".
        Default None
    mon_filepath: str, optional
        Path to the observed monitor dataset. Must end in .sgy. Required if save option
        is set to True. Default None.
    save: bool, optional
        Set to true if the DDWI dataset is to be saved in .sgy format. Default False
    save_path: str, optional
        Path to folder where DDWI must be saved. Default "./"
    verbose: bool, optional
        Set to true in order to verbose the steps of this function. Default 1

    Returns
    -------
    MonDiff: fullwaveqc.tools.SegyData
        Object outputted from fullwaveqc.tools.load function with the DDWI monitor data
        
    Notes
    -----
    mon_filepath must be given if save is set to True. Saving occurs by copying and modifying
    the monitor SEG-Y headers and traces.
    """

    # Set verbose level
    verbose_print = set_verbose(verbose)

    # Normalise traces of BasePred
    if normalise:
        verbose_print("\n" + str(datetime.datetime.now()) + " \t Normalising traces ...")
        BasePred = ampnorm(BaseObs, BasePred, verbose=verbose)

    # Compute double difference dataset
    verbose_print("\n" + str(datetime.datetime.now()) + " \t Computing double difference ...")
    MonDiff = copy.deepcopy(MonObs)
    for i, d in enumerate(MonDiff.data):
        MonDiff.data[i] = BasePred.data[i] + (MonObs.data[i] - BaseObs.data[i])
    verbose_print("\n" + str(datetime.datetime.now()) + " \t Double difference dataset calculated successfully!")

    # Assign new name to object
    if name is None:
        MonDiff.name = "DDWI-DATASET"
    else:
        MonDiff.name = name
    
    # Saving file
    if save:
        # Check destination path is valid for writing
        dstpath = save_path + name + ".sgy"
        try:
            with open(dstpath, 'w'):
                MonDiff.filepath = dstpath
        except IOError as x:
            raise IOError('error ', x.errno, ',', x.strerror, ", save_path invalid")

        # Check of mon_filepath is exists
        if mon_filepath is None:
            raise TypeError("mon_filepath must be given along with the save option")
        elif not os.path.isfile(mon_filepath):
            raise IOError("mon_pathfile invalid")

        with segyio.open(mon_filepath, ignore_geometry=True) as src:
            verbose_print("\n" + str(datetime.datetime.now()) + " \t Reading file %s..." % mon_filepath)
            spec = segyio.spec()
            spec.sorting = src.sorting
            spec.format = src.format
            spec.samples = src.samples
            spec.ilines = src.ilines
            spec.xlines = src.xlines
            spec.sorting = src.sorting
            spec.tracecount = src.tracecount

            # Create new file and copy all but trace data
            verbose_print("\n" + str(datetime.datetime.now()) + " \t Creating new file  at %s..." % dstpath)
            with segyio.create(dstpath, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin
                dst.header = src.header

                # Stack data
                verbose_print("\n" + str(datetime.datetime.now()) + " \t Stacking data ...")
                data = copy.deepcopy(MonDiff.data[0])
                for i in range(1, len(MonDiff.data)):
                    data = np.vstack((data, MonDiff.data[i]))

                # Update traces
                verbose_print("\n" + str(datetime.datetime.now()) + " \t Updating traces ...")
                dst.trace[:] = data
        verbose_print("\n" + str(datetime.datetime.now()) + " \t Double difference dataset saved!")

    return MonDiff


def bandpass(trace, flow, fhigh, forder, dt):
    """
    Band passes a trace using a Butter filter.

    Parameters
    ----------
    trace: numpy.array
        1D array containing the signal in time domain
    flow: float
        Low frquency to band pass (Hz)
    fhigh: float
        High frquency to band pass (Hz)
    forder: float
        Order of band pass filter, determines the steepnes of the transition band
    dt: float
        Time sampling of the signal

    Returns
    -------
    filtered: numpy.array
        1D array of same size as trace, with the filtered signal
    """

    # set up parameters for bandpass filtering
    nyq = 0.5 / dt  # nyquist frequency
    low = flow / nyq
    high = fhigh / nyq
    b, a = butter(forder, [low, high], btype='band')

    w, h = freqz(b, a, worN=2000)

    if max(abs(h) > 1.01):
        warnings.warn('Filter has values > 1. Will cause problems. Try reducing hte filter order')

    # determine if we are dealing with a trace or gather
    traceShape = np.shape(trace)
    nt = traceShape[0]
    if len(traceShape) == 2:
        nz = traceShape[1]
        filtered = np.zeros(shape=[nt, nz])
        for i in range(nz):
            filtered[:, i] = lfilter(b, a, trace[:, i])
    else:  # only 1 trace
        filtered = lfilter(b, a, trace)
    return filtered


def ampnorm(Obs, Pred, ref_trace=0, verbose=1):
    """
    Normalises the amplitude of a predicted dataset with an observed dataset by matching the values of their maximum
    amplitudes of a reference trace.

    Parameters
    ----------
    Obs: fullwaveqc.tools.SegyData
        Object outputted from fullwaveqc.tools.load function for the observed data
    Pred: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function for the predicted data.
        Requires same number of samples, sampling interval, shots and receiver positions
        as Obs
    ref_trace: int, optional
        Trace number to normalise each shot. For streamer data, this value should be
        0 in order to normalise the shot to the first arrival of the first trace. Default 0
    verbose: bool, optional
        Set to true to verbose the steps of this function. Default 1

    Returns
    -------
    PredNorm: fullwaveqc.tools.SegyData
        Object outputted from fullwaveqc.tools.load function with the normalise predicted data
    """

    # Set verbose
    verbose_print = set_verbose(verbose)

    # Copy original dataset and modify each trace of each shot
    PredNorm = copy.deepcopy(Pred)
    for i, d in enumerate(PredNorm.data):
        ratio = np.max(Obs.data[i][ref_trace]) / np.max(Pred.data[i][ref_trace])
        for j, trace in enumerate(d):
            PredNorm.data[i][j] = (PredNorm.data[i][j] * ratio)
        verbose_print(str(datetime.datetime.now()) + " \t Normalising shot %g\r" % i)
    sys.stdout.write(str(datetime.datetime.now()) + " \t All shots normalised")
    return PredNorm


def smooth_model(Model, strength=[1, 1], w=[None, None, None, None], slowness=True, name=None, save=False,
                 save_path="./", verbose=1):
    """
    Smoothes a model using scipy's gaussian filter in "reflect" mode.

    Parameters
    ----------
    Model: fullwaveqc.tools.Model
        Object outputted from fullwaveqc.tools.load function for a model. Must have a valid filepath attribute.
    strength: list[floats], optional
        [horizontal_smoothing_factor, vertical_smoothing_factor]. Default [1,1]
    w: list[ints], optional
        Windowing of the smoothing. Smoothing will be applied to the internal window created by the rectangle determined
        by [cells_from_top, cells_from bottom, cells_from_left, cells_from_right]. If None is given, then either the be
        ginning or end will of the model will be used. Default [None, None, None, None]
    slowness: bool, optional
        If True will smooth the slowness instead of velocities. Slowness defined
        as the reciprocal value of velocity and is commonly preferred in smoothing. Default True
    name: str, optional
        name of the new smoothed model. If None it will be inferred from Model.
        Default None
    save: bool, optional
        Set to true in order to save the model in .sgy format. Requires that Model
        has a valid filepath attribute. Default False
    save_path: str, optional
        Path to save the .sgy smoothed model. Default "./"
    verbose: bool, optional

    Returns
    -------
    SmoothModel: fullwaveqc.tools.Model
        Object as outputted from fullwaveqc.tools.load function containing the smoothed model

    """

    # Set verbose
    verbose_print = set_verbose(verbose)

    # make copy and edit model
    SmoothModel = copy.deepcopy(Model)
    if slowness:
        SmoothModel.data = 1. / SmoothModel.data
    # flip data so that windows make sense
    SmoothModel.data = np.flipud(SmoothModel.data)
    SmoothModel.data[w[0]:w[1], w[2]:w[3]] = gaussian_filter(SmoothModel.data[w[0]:w[1], w[2]:w[3]],
                                                             strength, mode="reflect")
    # unflip data to recover format
    SmoothModel.data = np.flipud(SmoothModel.data)

    if slowness:
        SmoothModel.data = 1. / SmoothModel.data

    if name is None:
        SmoothModel.name = SmoothModel.name + "-smooth"
    else:
        SmoothModel.name = name

    if save:
        dstpath = save_path + SmoothModel.name + ".sgy"

        with segyio.open(Model.filepath, ignore_geometry=True) as src:
            verbose_print("\n" + str(datetime.datetime.now()) + " \t Reading file %s..." % Model.filepath)
            spec = segyio.spec()
            spec.sorting = src.sorting
            spec.format = src.format
            spec.samples = src.samples
            spec.ilines = src.ilines
            spec.xlines = src.xlines
            spec.sorting = src.sorting
            spec.tracecount = src.tracecount

            # Create new file and copy all but trace data
            verbose_print("\n" + str(datetime.datetime.now()) + " \t Creating new file  at %s..." % dstpath)
            with segyio.create(dstpath, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin
                dst.header = src.header

                # Stack data
                verbose_print("\n" + str(datetime.datetime.now()) + " \t Stacking data ...")
                data = copy.deepcopy(SmoothModel.data[0])
                for i in range(1, len(SmoothModel.data)):
                    data = np.vstack((data, SmoothModel.data[i]))
                data = np.fliplr(data.T)

                # Update traces
                verbose_print("\n" + str(datetime.datetime.now()) + " \t Updating model ...")
                dst.trace[:] = data
        verbose_print("\n" + str(datetime.datetime.now()) + " \t Smoothed model saved!")

    return SmoothModel
