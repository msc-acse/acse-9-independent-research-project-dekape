import numpy as np
import matplotlib.pyplot as plt
import warnings

def functional(filepath, name=None, plot=True, save=False, save_path="./FIGURES/"):

    # If name not given, get it from filename
    if name is not None:
        name = name
    else:
        name = str(filepath).split("/")[-1].split('.')[0]

    # Create list to store functional values
    functional = []

    # Read file and search for line with "global functional" in it
    with open(filepath) as joblog:
        for line in joblog:
            if "global functional" in line:
                functional.append(float(line.split()[5]))

    functional = np.array(functional)
    iter_all = np.arange(0, len(functional), 1)

    if plot:
        figure, ax = plt.subplots(1, 1)
        figure.set_size_inches(10.5, 10.5)
        ax.loglog(iter_all, functional, 'k-')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Global Functional Value")
        ax.set_title(name + "-Convergence")
        ax.grid(True, which="minor", axis='both')

    if save:
        plt.savefig(save_path + name + "-CONVERGENCE.png", dpi=300)

    else:
        plt.show()

    return iter_all, functional


def gradient():
    return


def steplen(filepath, name=None, plot=True, save=False, save_path="./FIGURES/"):

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
    iter_all = np.arange(0, len(steplenarr), 1)

    if plot:
        figure, ax = plt.subplots(1, 1)
        figure.set_size_inches(10.5, 10.5)
        ax.plot(iter_all, steplenarr, 'k.-')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Global Step Length")
        ax.set_title(name + "-Step Length")
        ax.grid(True)

    if save:
        plt.savefig(save_path + name + "-CONVERGENCE.png", dpi=300)

    else:
        plt.show()

    return iter_all, steplen
