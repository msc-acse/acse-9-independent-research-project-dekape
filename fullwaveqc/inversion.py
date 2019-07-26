#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import warnings


def functional(filepath, name=None, plot=False):
    """
    Retrieves and plots the global functional value of an inversion run with Fullwave3D as  recorded in the job log
    file,
    :param  filepath:    (str)      path to <PROJECT_NAME>_job,log file
    :param  name:        (str)      name of the project. If not given, will be inferred from filename. Default: None
    :param  plot:        (bool)     if true will produce a plot of the functional values vs iterations. Default: False
    :return iter_all     (np.array) array of iteration numbers of the run of size (number_of_iterations, 1)
    :return func         (np.array) array of functional of size (number_of_iterations, 1)
    """

    # If name not given, get it from filename
    if name is not None:
        name = name
    else:
        name = str(filepath).split("/")[-1].split('.')[0]

    # Create list to store functional values
    func = []

    # Read file and search for line with "global functional" in it
    with open(filepath) as joblog:
        for line in joblog:
            if "global functional" in line:
                func.append(float(line.split()[5]))

    func = np.array(func)
    iter_all = np.arange(0, len(func), 1) + 1

    if plot:
        figure, ax = plt.subplots(1, 1)
        figure.set_size_inches(9.5, 9.5)
        ax.plot(iter_all, func, 'k-')
        # ax.set_yscale('log')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Global Functional Value")
        ax.set_title(name + "-Convergence")
        ax.grid(True)
        plt.show()

    return iter_all, func


def steplen(filepath, name=None, plot=False):
    """
    Retrieves and plots the step length value of an inversion run with Fullwave3D as recorded in the job log
    file,
    :param  filepath:    (str)      path to <PROJECT_NAME>_job,log file
    :param  name:        (str)      name of the project. If not given, will be inferred from filename. Default: None
    :param  plot:        (bool)     if true will produce a plot of the step length values vs iterations. Default: False
    :return iter_all     (np.array) array of iteration numbers of the run of size (number_of_iterations, 1)
    :return steplenarr   (np.array) array of step length values of size (number_of_iterations, 1)
    """

    # If name not given, get it from filename
    if name is not None:
        name = name
    else:
        name = str(filepath).split("/")[-1].split('.')[0]

    # Create list to store functional values
    steplenarr = []

    # Read file and search for line with "global functional" in it
    count = 0
    with open(filepath) as joblog:
        for line in joblog:
            count += 1
            if "step-factor" in line:
                try:
                    steplenarr.append(float(line.split(",")[0].split("=")[-1]))
                except ValueError:
                    try:
                        steplenarr.append(float(line.split(",")[0].split("=")[-1].split(" ")[0]))
                    except ValueError:
                        warnings.warn("Not able to read step length in line %g" % count)

    steplenarr = np.array(steplenarr)
    iter_all = np.arange(0, len(steplenarr), 1) + 1

    # Get all <= 0 values
    id_neg = np.where(steplenarr <= 0.)

    if plot:
        figure, ax = plt.subplots(1, 1)
        figure.set_size_inches(9.5, 9.5)
        ax.plot(iter_all, steplenarr, 'ko-')
        ax.plot(iter_all[id_neg], steplenarr[id_neg], 'ro')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Global Step Length")
        ax.set_title(name + "-Step Length")
        ax.grid(True)
        plt.show()

    return iter_all, steplenarr


def gradient():
    return
