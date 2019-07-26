#!/usr/bin/env python

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
    def __init__(self, filepath, name=None):
        if name is not None:
            self.name = name
        else:
            self.name = filepath.split("/")[-1].split('.')[0]
        self.filepath = filepath
        self.src_pos = None
        self.rec_pos = None
        self.nsrc = -1
        self.nrec = -1
        self.data = None
        self.src_z = None
        self.rec_z = None
        self.samples = None
        self.dt = None
        self.dsrc = None
        self.drec = None
        self._type = "data"

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
        self._type = "model"

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


def load(filepath, model, scale=1, verbose=1):
    """
    Loads a segy file into a SegyData or Model class, used for the other functionalities of the fullwaveqc package.
    Accepts 2D segy files and of the same sampling interval for all shots and segy model files with square cells

    :param    filepath: (str)   path to segy file. Must end in .sgy
    :param    model:    (bool)  Set true to load a Model and false to load a SegyData object
    :param    scale:    (float) Value to multiply the data content of the files. Default 1
    :param    verbose:  (bool)  Set to true in order to verbose the steps of this loading function. Default True
    :return::  fullwaveqc.tools.SegyData or fullwave,tools.Model object
    """

    if not model:
        # Create Data object
        segy = SegyData(filepath)

        # Open datafile with segyio
        if verbose:
            print(datetime.datetime.now(), " \t Opening file %s..." % filepath)
        with segyio.open(filepath, mode='r+', ignore_geometry=True) as segyfile:
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
            print(datetime.datetime.now(), " \t Successfully loaded model of size %g x %g!" % (model.nz, model.nx))
        model.maxvp = np.max(data)
        model.minvp = np.min(data)

        return model


def rm_empty_traces(filename, scale=1, verbose=1):
    """
    Removes all-zero traces of a segy file and saves the copy of the clean file in the same directory

    :param    filename: (str)   path to segy file, must end in .sgy
    :param    scale:    (float) Value to multiply the data content of the files. Default 1
    :param    verbose:  (bool)  Set to true in order to verbose the steps of this loading function. Default True
    :return::  None
    """

    #
    dstpath = filename[:-4]+"-CLEAN.sgy"

    if verbose:
        print(datetime.datetime.now(), " \t Opening file %s..." % filename)
    with segyio.open(filename, mode='r', ignore_geometry=True) as segyfile:
        if verbose:
            print(datetime.datetime.now(), " \t Reading trace data...")
        data = np.asarray([trace * 1.0 for trace in segyfile.trace])*scale  # (number of traces, number of samples)

        if verbose:
            print(datetime.datetime.now(), " \t Reading headers...")
        header = []
        for i in range(0, data.shape[0]):
            header.append(segyfile.header[i])  # list of dictionaries

        if verbose:
            print(datetime.datetime.now(), " \t Collecting header parameters...")
        newspec = segyio.spec()
        newspec.ilines = segyfile.ilines
        newspec.xlines = segyfile.ilines
        newspec.samples = segyfile.samples
        newspec.sorting = segyfile.sorting
        newspec.format = segyfile.format

    if verbose:
        print(datetime.datetime.now(), " \t Cleaning traces and headers ...")
    newdata = data[~np.all(data == 0, axis=1)]
    newheader = []
    for i in range(0, data.shape[0]):
        if not (data[i, :] == 0).all():
            newheader.append(header[i])
    assert newdata.shape[0] == len(newheader)

    if verbose:
        print(datetime.datetime.now(), " \t Removed %g empty traces!" % (data.shape[0] - newdata.shape[0]))
    newspec.tracecount = newdata.shape[0]

    if verbose:
        print(datetime.datetime.now(), " \t Writing trace and headers to new file at %s..." % dstpath)
    with segyio.create(dstpath, newspec) as newsegy:
        newsegy.trace[:] = newdata
        for i in range(0, newdata.shape[0]):
            newsegy.header[i] = newheader[i]

    if verbose:
        print(datetime.datetime.now(), " \t Clean file created successfully!")

    return None


def ddwi(MonObs, BaseObs, BasePred, normalise=True, name=None, mon_filepath=None, save=False, save_path="./",
         verbose=1):
    """
    Creates the monitor dataset used as the observed dataset for the Double Difference Waveform Inversion.
    The DDWI monitor dataset is given by: u_mon = d_mon - d_base + u_base
    Where d_mon and d_base are the observed monitor and baseline datasets, respectively, and u_base is the dataset from
    a predicted baseline inversion model.

    :param  MonObs:        (SegyData)   object outputted from fullwaveqc.tools.load function for the true monitor data
    :param  BaseObs:       (SegyData)   object outputted from fullwaveqc.tools.load function for the true baseline data.
                                        Requires same number of samples, sampling interval, shots and receiver positions
                                        as MonObs
    :param  BasePred:      (SegyData)   object outputted from fullwaveqc.tools.load function for the predicted baseline
                                        data. Requires same number of samples, sampling interval, shots and receiver
                                        positions as MonObs.
    :param  normalise:     (bool)       if true will normalise the amplitude of the predicted baseline dataset to match
                                        the first trace of the observed dataset. Does so shot by shot. Default True
    :param  name:          (str)        name to give the DDWI dataset, if None will give the name "DDWI-DATASET".
                                        Default None
    :param  mon_filepath:  (str)        path to the observed monitor dataset. Must end in .sgy. Required if save option
                                        is set to True. Default None.
    :param  save:          (bool)       Set to true if the DDWI dataset is to be saved in .sgy format. Default False
    :param  save_path:     (str)        Path to folder where DDWI must be saved. Default "./"
    :param  verbose:       (bool)       Set to true in order to verbose the steps of this function. Default 1
    :return: MON_DIFF       (SegyData)   object outputted from fullwaveqc.tools.load function with the DDWI monitor data
    """

    # error handling: check dst path is valid
    # optimise: stacking

    # Normalise traces of BasePred
    if normalise:
        if verbose:
            print(datetime.datetime.now(), " \t Normalising traces ...")
        BasePred = ampnorm(BaseObs, BasePred)

    # Compute double difference dataset
    if verbose:
        print(datetime.datetime.now(), " \t Computing double difference ...")
    MON_DIFF = copy.deepcopy(MonObs)

    for i, d in enumerate(MON_DIFF.data):
        MON_DIFF.data[i] = BasePred.data[i] + (MonObs.data[i] - BaseObs.data[i])
    if verbose:
        print(datetime.datetime.now(), " \t Double difference dataset calculated successfully!")

    # Assign new name to object
    if name is None:
        MON_DIFF.name = "DDWI-DATASET"
    else:
        MON_DIFF.name = name

    if save:
        if mon_filepath is None:
            raise NameError

        dstpath = save_path + name

        # Read monitor file and store SEG-Y information
        if verbose:
            print(datetime.datetime.now(), " \t Opening file %s..." % mon_filepath)
        with segyio.open(mon_filepath, mode='r', ignore_geometry=True) as segyfile:
            if verbose:
                print(datetime.datetime.now(), " \t Reading trace data...")
            data = np.asarray(
                [trace * 1.0 for trace in segyfile.trace])  # data in format (number of traces, number of samples)

            if verbose:
                print(datetime.datetime.now(), " \t Reading headers...")
            header = []
            for i in range(0, data.shape[0]):
                header.append(segyfile.header[i])  # list of dictionaries

            if verbose:
                print(datetime.datetime.now(), " \t Collection segy header parameters...")
            newspec = segyio.spec()
            newspec.ilines = segyfile.ilines
            newspec.xlines = segyfile.ilines
            newspec.samples = segyfile.samples
            newspec.sorting = segyfile.sorting
            newspec.format = segyfile.format

        if verbose:
            print(datetime.datetime.now(), " \t Initiating saving...")

        # Stack data
        if verbose:
            print(datetime.datetime.now(), " \t Stacking data ...")
        newdata = copy.deepcopy(MON_DIFF.data[0])
        for i in range(1, len(MON_DIFF.data)):
            newdata = np.vstack((newdata, MON_DIFF.data[i]))
        assert (newdata.shape == data.shape)

        # Copy over trace headers
        if verbose:
            print(datetime.datetime.now(), " \t Updating traces ...")
        newheader = header
        newspec.tracecount = newdata.shape[0]

        # Write to new file
        if verbose:
            print(datetime.datetime.now(), " \t Writing trace and headers to new file %s..." % dstpath)
        with segyio.create(dstpath, newspec) as newsegy:
            newsegy.trace[:] = newdata
            for i in range(0, newdata.shape[0]):
                newsegy.header[i] = newheader[i]

        if verbose:
            print(datetime.datetime.now(), " \t Double difference dataset saved!")

    return MON_DIFF


def bandpass(trace, flow, fhigh, forder, dt):
    """
    Band passes a trace using a Butter filter.

    :param  trace:     (np.array) 1D array containing the signal in time domain
    :param  flow:      (float)    low frquency to band pass
    :param  fhigh:     (float)    high frequency to band pass
    :param  forder:    (int)      order of band pass filter, determines the steepnes of the transition band
    :param  dt:        (float)    Time sampling of the signal
    :return: filtered   (np.array) 1D array of same size as trace, with the filtered signal
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

    :param  Obs:         (SegyData)   object outputted from fullwaveqc.tools.load function for the observed data
    :param  Pred:        (SegyData)   object outputted from fullwaveqc.tools.load function for the predicted data.
                                      Requires same number of samples, sampling interval, shots and receiver positions
                                      as Obs
    :param  ref_trace:   (int)        trace number to each normalise each shot. For streamer data, this value should be
                                      0 in order to normalise it to the first arrival trace. Default 0
    :param  verbose:     (bool)       Set to true to verbose the steps of this function. Default 1
    :return: PredNorm     (SegyData)   object outputted from fullwaveqc.tools.load function with the normalise predicted
                                      data
    """
    PredNorm = copy.deepcopy(Pred)
    for i, d in enumerate(PredNorm.data):
        ratio = np.max(Obs.data[i][ref_trace]) / np.max(Pred.data[i][ref_trace])
        for j, trace in enumerate(d):
            PredNorm.data[i][j] = (PredNorm.data[i][j] * ratio)
        if verbose:
            sys.stdout.write(str(datetime.datetime.now()) + " \t Normalising shot %g\r" % i)
    print(str(datetime.datetime.now()) + " \t All shots normalised")
    return PredNorm


def smooth_model(Model, strength=[1, 1], name=None, save=False, save_path="./"):
    """
    Smoothes a model using scipy's gaussian filter in "reflect" mode.

    :param  Model:       (Model)           object outputted from fullwaveqc.tools.load function for a model
    :param  strength:    (list of floats)  [horizontal_smoothing_factor, vertical_smoothing_factor]. Default [1,1]
    :param  save:        (bool)            set to true in order to save the model in .sgy format. Default False
    :param  save_path:   (str)             path to save the .sgy smoothed model. Default "./"
    :param  name:        (str)             name of the new smoothed model. If None it will be inferred from Model.
                                           Default None
    :return: SmoothModel: (Model)           object as outputted from fullwaveqc.tools.load function containing the
                                           smoothed model
    """
    SmoothModel = copy.deepcopy(Model)
    SmoothModel.data = gaussian_filter(SmoothModel.data, strength, mode="reflect")
    if name is None:
        SmoothModel.name = SmoothModel.name + "-smooth"
    else:
        SmoothModel.name = name

    if save:
        print("Feature not yet implemented!")
    return SmoothModel


# def save(Obj, save_path, verbose=1):
#
#     if Obj._type == "data":
#         newdata = copy.deepcopy(Obj.data[0])
#         for i in range(1, len(Obj.data)):
#             newdata = np.vstack((newdata, Obj.data[i]))
#
#         samples = np.arange(0, Obj.samples[0], 1) * Obj.dt[0]
#         for i in range(1, len(Obj.samples)):
#             samples = np.hstack((samples, np.arange(0, Obj.samples[i] - 1, 1) * Obj.dt[i] + samples[-1] + Obj.dt[i]))
#
#         # create segyio spec object
#         newspec = segyio.spec()
#         newspec.ilines = None
#         newspec.xlines = None
#         newspec.samples = samples
#         newspec.sorting = None
#         newspec.tracecount = newdata.shape[0]
#         newspec.format = 5  # "4-byte IEEE float"
#         print(newspec.tracecount)
#
#         with segyio.create(save_path, newspec) as newsegy:
#             newsegy.trace[:] = newdata
#
#     else:
#         a = 1
#
#     return None

