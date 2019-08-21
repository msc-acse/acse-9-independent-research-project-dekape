#!/usr/bin/env python
# Deborah Pelacani Cruz
# https://github.com/dekape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as an
from IPython.core.display import display, HTML
import matplotlib
import copy
import fullwaveqc.tools as tools
import sys
import datetime
import os

# Setting matplotlib plotting parameters
plt.rcParams.update({'font.size': 14})
matplotlib.rcParams['animation.embed_limit'] = 2**128


class AnimateImshow:
    def __init__(self, arrays, dt=1, vmin=None, vmax=None, cmap=plt.cm.jet):
        """
        Array is a list of 2d arrays or a 3d array with time on axis=0.

        """
        # setup data
        self.arr = arrays
        self.Nt = len(arrays)
        self.dt = dt

        # setup the figure
        self.fig = plt.figure()
        self.fig.set_size_inches(18.5, 10.5)
        self.im = plt.imshow(arrays[0], animated=True, cmap=cmap, vmin=vmin, vmax=vmax)
        self.ax = plt.gca()
        self.ax.invert_yaxis()
        self.ax.axis("off")
        self.text = self.ax.text(0.1, 0.1, 'Time: 0', color='w')

        # settings
        self.delay = 10
        self.title = "title"
        self.xlabel = "xlabel"
        self.ylabel = "ylabel"

        # run the animation

    def run(self, html=False):
        self.ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)

        self.ani = an.FuncAnimation(self.fig, self.update, frames=range(self.Nt),
                                    interval=self.delay, blit=True)

        if html:  # use this if in ipython/jupyter notebooks
            plt.close(self.fig)
            self.jshtml = self.ani.to_jshtml()
            display(HTML(self.jshtml))

    def save_html(self, filename="output.html"):
        if not hasattr(self, 'jshtml'):
            self.jshtml = self.ani.to_jshtml()
        f = open(filename, 'w')
        f.write(self.jshtml)
        f.close()

    def update(self, f):
        self.im.set_array(self.arr[f])
        self.text.set_text('Time: %i' % f * self.dt)
        return self.im, self.text


def amplitude(SegyData, shot=1, cap=0., levels=50, vmin=None, vmax=None, cmap=plt.cm.seismic,
              xstart=0, xend=None, wstart=0, wend=None, save=False, save_path="./"):
    """
    Plots the amplitude map of a SegyData object in timesamples (dt units) vs receiver index.
    Uses properties of matplotlib.contourf.

    Parameters
    ----------
    SegyData: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function for the segy data
    shot: int, optional
        shot number to visualise. Count starts from 1. Default 1
    cap: float, optional
        absolute value of amplitudes to cap, shown as 0. See matplotlib.pyplot.contourf parameters. Default 0
    levels: int, optional
        amount of levels in the contour plot. See matplotlib.pyplot.contourf parameters. Default 50
    vmin: float, optional
        min val of color plot scale. See matplotlib.pyplot.contourf parameters. Default None.
    vmax: float, optional
        max val of color plot scale. See matplotlib.pyplot.contourf parameters. Default None.
    cmap: matplotlib.pyplot.cm
        colormap. Default plt.cm.seismic
    xstart: int, optional
        first receiver index to plot. Default 0.
    xend: int, optional
        last receiver index to plot. Default None
    wstart: int, optional
        first timesample in time units to plot. Default 0
    wend: int, optional
        last timesample in time units to plot. Default None
    save: bool, optional
        set to true in order to save the plot in png 300dpi. Default False
    save_path: str, optional
        path to save the plot. Default "./"

    Returns
    -------
    None

    """
    if save:
        if save_path[-1] != "/":
            save_path = save_path + "/"
        if not os.path.isdir(save_path):
            raise Exception("Save path is invalid")

    try:
        # Get dt from SegyData object
        dt = SegyData.dt[shot-1]

        # Transform ms window to index
        wstart = int(wstart / dt)
        if wend is not None:
            wend = int(wend / dt)
            if wend > SegyData.samples[shot-1]:
                wend = SegyData.samples[shot - 1]
        else:
            wend = SegyData.samples[shot-1]

        # Getting last receiver is xend is not defined
        if xend is None:
            xend = SegyData.nrec[shot-1]
        elif xend > SegyData.nrec[shot-1]:
            xend = SegyData.nrec[shot-1]

        # Check inputs are sensible
        if wstart < 0 or xstart < 0:
            raise ValueError("wstart and xstart must be zero or positive numbers")
        if wend - 2 <= wstart:
            raise ValueError("Window too small or negative. Increase the wend wstart gap.")
        if xend <= xstart:
            raise ValueError("xend must be larger than xstart")

        # Get data from SegyData object
        data = SegyData.data[shot-1][xstart:xend, wstart:wend]

        # Cap amplitude values
        data[np.where(np.logical_and(data >= (-1) * cap, data <= cap))] = 0.

        # Create array axes for plotting
        x = np.arange(xstart, xend, 1)
        y = np.linspace(wend, wstart, data.shape[1]) * dt

        # Plot figure
        figure, ax = plt.subplots(1, 1)
        figure.set_size_inches(18.5, 10.5)
        ax.contourf(x, y, np.rot90(data), cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(SegyData.name + " Shot {}".format(shot))
        ax.set_ylabel("Time (ms)")
        ax.set_xlabel("Rec x")

        # Formatting extent of colorbar
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(data)
        m.set_clim(vmin, vmax)
        plt.colorbar(m)

        # Saving option
        if save:
            plt.savefig(save_path + SegyData.name + ".png", dpi=300)
        else:
            plt.show()

    except AttributeError as e:
        raise type(e)("SegyData must be a fullwaveqc.tools.SegyData instance.")

    return None


def interamp(SegyData1, SegyData2, shot=1, shot2=None, n_blocks=2, cap=0., levels=50, vmin=None, vmax=None,
             cmap=plt.cm.seismic, wstart=0, wend=None, xstart=0, xend=None, save=False, save_path="./"):
    """
    Plots the interleaving amplitude map of a SegyData1 and SegyData2 objects in timesamples vs receiver index.
    Uses properties of matplotlib.contourf.

    Parameters
    ----------
    SegyData1: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function for the segy data
    SegyData2: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function for the segy data
        Requires same number of samples, sampling interval, shots and receiver
        positions as SegyData1
    shot: int, optional
        shot number to visualise SegyData1. Count starts from 1. Default 1
    shot2: int, optional
        shot number to visualise SegyData2. Count starts from 1. If None, will .
        be read the same as 'shot'. Default None
    n_blocks: int, optional
        Number of total blocks in the interleaving space
    levels: int, optional
        amount of levels in the contour plot. See matplotlib.pyplot.contourf parameters. Default 50
    cap: float, optional
        absolute value of amplitudes to cap, shown as 0. Default 0.
    vmin: float, optional
        min val of color plot scale. See matplotlib.pyplot.contourf parameters. Default None.
    vmax: float, optional
        max val of color plot scale. See matplotlib.pyplot.contourf parameters. Default None.
    cmap: matplotlib.pyplot.cm
        colormap. Default plt.cm.seismic
    xstart: int, optional
        first receiver index to plot. Default 0.
    xend: int, optional
        last receiver index to plot. Default None
    wstart: int, optional
        first timesample in time units to plot. Default 0
    wend: int, optional
        last timesample in time units to plot. Default None
    save: bool, optional
        set to true in order to save the plot in png 300dpi. Default False
    save_path: str, optional
        path to save the plot. Default "./"


    Returns
    -------
    None

    """
    if not (isinstance(SegyData1, tools.SegyData) and isinstance(SegyData2, tools.SegyData)):
        raise AttributeError("SegyData must be a fullwaveqc.tools.SegyData instance.")

    data1 = SegyData1.data[shot-1]
    if shot2 is not None:
        data2 = SegyData2.data[shot2-1]
    else:
        data2 = SegyData2.data[shot-1]

    assert (data1.shape == data2.shape)

    nx = data1.shape[0]
    block_len = int(nx / n_blocks)
    data_inter = np.zeros_like(data1)

    init = 0
    for i in range(0, n_blocks, 2):
        data_inter[init:block_len * (i + 1), ] = data1[init:block_len * (i + 1), ]
        data_inter[init + block_len:block_len * (i + 2), :] = data2[init + block_len:block_len * (i + 2), :]
        init = init + 2 * block_len

    InterData = copy.deepcopy(SegyData1)
    InterData.data[shot-1] = data_inter
    InterData.name = SegyData2.name + "-" + SegyData1.name + "-INTERAMP"
    amplitude(InterData, shot=shot, cap=cap, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap, save=save,
              wstart=wstart, wend=wend, xstart=xstart, xend=xend, save_path=save_path)
    del InterData
    return None


def wiggle(SegyData, shot=1, scale=5, skip_trace=20, skip_time=0, wstart=0., wend=None, xstart=0, xend=None,
           delay_samples=0, save=False, save_path="./"):
    """
    Plots the wiggle trace map of a SegyData object in timesamples vs receiver index.

    Parameters
    ----------
    SegyData: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function for the segy data
    shot: int, optional
        shot number to visualise SegyData1. Count starts from 1. Default 1
    scale: float, optional
        value to scale the amplitude of the wiggles for visualisation only. Default 1
    skip_trace: int, optional
        Number of traces to skip. Default 20.
    skip_time: int, optional
        Number of time samples to skip. Default 0
    xstart: int, optional
        first receiver index to plot. Default 0.
    xend: int, optional
        last receiver index to plot. Default None
    wstart: int, optional
        first timesample in time units to plot. Default 0
    wend: int, optional
        last timesample in time units to plot. Default None
    delay_samples: int, optional
        number of time samples to delay the signal. Will pad the signal with 0s at the
        beginning of each trace. Default 0.
    save: bool, optional
        set to true in order to save the plot in png 300dpi. Default False
    save_path: str, optional
        path to save the plot. Default "./"

    Returns
    -------
    None
    """

    # Get data from SegyData object
    data = SegyData.data[shot-1]

    # Stack delay to data
    delay = np.zeros((data.shape[0], delay_samples))
    data = np.hstack((delay, data))

    offsets = np.arange(0, data.shape[0], skip_trace+1)
    times = np.arange(0, data.shape[1], skip_time + 1) * SegyData.dt[shot-1]

    figure, ax = plt.subplots(1, 1)
    figure.set_size_inches(18.5, 10.5)
    ax.invert_yaxis()

    for i in offsets:
        x = scale*data[i] + i
        ax.plot(x, times, '-k')
        ax.fill_betweenx(times, i, x, where=(x > i), color='k')

    ax.set_title(SegyData.name + " Shot {}".format(shot))
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Rec x")
    ax.grid(True, axis='y')

    if wend is None:
        wend = times[-1]
    if xend is None:
        xend = x[-1]
    ax.set_ylim(wend, wstart)
    ax.set_xlim(xstart, xend)

    if save:
        plt.savefig(save_path + SegyData.name + "-WIGGLE.png", dpi=300)
    else:
        plt.show()

    return None


def interwiggle(SegyData1, SegyData2, shot=1, shot2=None, overlay=0, scale=5, skip_trace=20, skip_time=0,
                delay_samples=0, wstart=0, wend=None, xstart=0, xend=None, label1="SegyData1", label2="SegyData2",
                save=False, save_path="./"):
    """
    Plots the interleaving wiggle trace map of a SegyData1 and SegyData2 objects in timesamples vs receiver index.

    Parameters
    ----------
    SegyData1: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function for the segy data
    SegyData2: fullwaveqc.tools.SegyData
        object outputted from fullwaveqc.tools.load function for the segy data
        Requires same number of samples, sampling interval, shots and receiver
        positions as SegyData1
    shot: int, optional
        shot number to visualise SegyData1. Count starts from 1. Default 1
    shot2: int, optional
        shot number to visualise SegyData2. Count starts from 1. If None, will
        be read the same as 'shot'. Default None
    overlay: int, optional
        If +ve, traces will be overlaid in different colours instead of being
        displayed side by side. Overlay=1 overlays with all filled colors, overlay=2
        overlays with SegyData2 as filled colors and SegyData1 as conotur. Default 0
    scale: float, optional
        value to scale the amplitude of the wiggles for visualisation only. Default 1
    skip_trace: int, optional
        Number of traces to skip. Default 20.
    skip_time: int, optional
        Number of time samples to skip. Default 0
    delay_samples: int, optional
        number of time samples to delay the signal. Will pad the signal with 0s at the
        beginning of each trace. Default 0.
    xstart: int, optional
        first receiver index to plot. Default 0.
    xend: int, optional
        last receiver index to plot. Default None
    wstart: int, optional
        first timesample in time units to plot. Default 0
    wend: int, optional
        last timesample in time units to plot. Default None
    label1: str, optional
        Label for SegyData1. Default SegyData1
    label2: str, optional
        Label for SegyData2. Default SegyData2
    save: bool, optional
        set to true in order to save the plot in png 300dpi. Default False
    save_path: str, optional
        path to save the plot. Default "./"

    Returns
    -------
    None
    """

    data1 = SegyData1.data[shot-1]
    if shot2 is not None:
        data2 = SegyData2.data[shot2-1]
    else:
        data2 = SegyData2.data[shot-1]

    assert (data1.shape == data2.shape)

    # Stack delay to data
    delay = np.zeros((data1.shape[0], delay_samples))
    data1 = np.hstack((delay, data1))
    data2 = np.hstack((delay, data2))

    nx = data1.shape[0]
    nz = data1.shape[1]

    # Create x and y axis (offset  and time)
    offsets = np.arange(0, nx, skip_trace + 1)
    times = np.arange(0, nz, skip_time + 1) * SegyData1.dt[shot-1]

    # Create figure
    figure, ax = plt.subplots(1, 1)
    figure.set_size_inches(18.5, 10.5)
    ax.invert_yaxis()

    # Overlay traces
    if overlay:
        for i in offsets:
            x2 = scale * data2[i] + i
            ax.plot(x2, times, '-w', linewidth=0)
            ax.fill_betweenx(times, i, x2, where=(x2 > i), color='r')
            ax.fill_betweenx(times, i, x2, where=(x2 < i), color='b')
            ax.fill_betweenx(times, i, x2, where=(x2 < i), color='b')

            x1 = scale * data1[i] + i
            ax.plot(x1, times, '-k')
            if overlay == 1:
                ax.fill_betweenx(times, i, x1, where=(x1 > i), color='k')
                ax.fill_betweenx(times, i, x1, where=(x1 < i), color='k')

    # Interleave traces
    else:
        for i in offsets:
            x2 = scale*data2[i] + i + int(skip_trace/2)
            ax.plot(x2, times, '-r')
            ax.fill_betweenx(times, i + int(skip_trace/2), x2, where=(x2 > i + int(skip_trace/2)), color='r')

            x1 = scale*data1[i] + i
            ax.plot(x1, times, '-k')
            ax.fill_betweenx(times, i, x1, where=(x1 > i), color='k')

    custom_lines = [Line2D([0], [0], color="k", lw=1),
                    Line2D([0], [0], color="r", lw=1)]
    ax.set_title(SegyData2.name + "-" + SegyData1.name + "-INTERWIGGLE")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Rec x")
    ax.set_ylim(wend, wstart)
    ax.set_xlim(xstart, xend)
    ax.grid(True, axis='y')
    ax.legend(custom_lines, [label1, label2], loc=7, bbox_to_anchor=(1.1, 0.5))

    if save:
        plt.savefig(save_path + SegyData1.name + "-INTERWIGGLE"+".png", dpi=300)
    else:
        plt.show()

    return None


def vpmodel(Model, cap=0., levels=200, vmin=None, vmax=None, cmap=plt.cm.jet, units="m/s",
            save=False, save_path="./"):
    """
    Plots the P-velocity amplitude map of a Model object in depth and lateral offset.
    Uses properties of matplotlib.contourf.

    Parameters
    ----------
    Model: fullwavqc.tools.Model
        object outputted from fullwaveqc.tools.load function for a model
    levels: int, optional
        amount of levels in the contour plot. See matplotlib.pyplot.contourf parameters. Default 200
    cap: float, optional
        absolute value of amplitudes to cap, shown as 0. Default 0.
    vmin: float, optional
        min val of color plot scale. See matplotlib.pyplot.contourf parameters. Default None.
    vmax: float, optional
        max val of color plot scale. See matplotlib.pyplot.contourf parameters. Default None.
    cmap: matplotlib.pyplot.cm
        colormap. Default plt.cm.jet
    units: str, optional
        units of amplitudes to show in colorbar. Default m/s
    save: bool, optional
        set to true in order to save the plot in png 300dpi. Default False
    save_path: str, optional
        path to save the plot. Default "./"

    Returns
    -------
    None
    """

    # Get data from Model
    data = Model.data
    dx = Model.dx
    name = Model.name

    # Cap values
    data[np.where(np.logical_and(data >= (-1)*cap, data <= cap))] = 0.

    # Plot figure
    figure, ax = plt.subplots(1, 1)
    figure.set_size_inches(16.5, 9.5)

    # Contour plot and formatting axis
    x = np.arange(0, data.shape[1], 1)*dx
    y = np.arange(data.shape[0], 0, -1)*dx
    ax.contourf(x, y, data, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_title(name)
    ax.set_ylabel("Depth")
    ax.set_xlabel("X")

    # Format limits of colorbar
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(data)
    m.set_clim(vmin, vmax)
    cbar = plt.colorbar(m)
    cbar.set_label(units)

    if save:
        plt.savefig(save_path + name + ".png", dpi=300)
    else:
        plt.show()

    return None


def vpwell(Model, pos_x, TrueModel=None,  plot=True):
    """
    Retrieves the Vp well profile from a Model at specified locations

    Parameters
    ----------
    Model: fullwaveqc.SegyData.Model
        object outputted from fullwaveqc.tools.load function for a model
    pos_x: list(ints)
        lateral distances to which retrieve the well data
    TrueModel: fullwaveqc.SegyData.Model, optional
        object outputted from fullwaveqc.tools.load function for the true model. Default None
    plot: bool, optional
        Plots the profiles when set to true.

    Returns
    -------
    wells: list[numpy.array]
        Well values with depth
    true_wells:list[numpy.array]
        Only if TrueModel is given. True well values with depth
    rmses:list[numpy.array]
        Only if TrueModel is given. RMS errors between true well and predicted well values.

    """

    # Get model data
    pred = np.flipud(Model.data)

    # Get true model
    if TrueModel is not None:
        obs = np.flipud(TrueModel.data)
    else:
        obs = None

    # Create list to store retrieved well data
    n = len(pos_x)
    wells = []
    true_wells = []
    rmses = []
    ypreds = []
    ytrues = []

    # For each well positiong
    for i in range(0, n):
        # Get well data for predicted model
        grid_pos = int(pos_x[i]/Model.dx)
        well = pred[:, grid_pos]
        wells.append(well)

        # Get Well data for observed model
        if obs is not None:
            true_grid_pos = int(pos_x[i]/TrueModel.dx)
            true_well = obs[:, true_grid_pos]
            true_wells.append(true_well)

        # Get y axis of predicted signal for interpolation
        ypred = np.arange(0, well.shape[0], 1) * Model.dx
        ypreds.append(ypred)

        # Calculate RMSE, interpolate predicted data first
        if obs is not None:
            ytrue = np.arange(0, true_well.shape[0], 1) * TrueModel.dx
            ytrues.append(ytrue)
            well_interp = np.interp(ytrue, ypred, well)
            rmse = np.sqrt(np.mean((well_interp - true_well)**2))
            rmses.append(rmse)
        else:
            rmses.append(0.0)

    if plot:
        figure, axs = plt.subplots(1, n)
        figure.set_size_inches(9 * n, 20)
        for i in range(0, n):
            if n > 1:
                ax = axs[i]
            else:
                ax = axs

            ax.plot(wells[i], ypreds[i], 'b', label="Predicted")
            if obs is not None:
                ax.plot(true_wells[i], ytrues[i], 'k', label="Observed")
                ax.legend(loc="best")
            ax.set_title(Model.name+r" x=%i, RMSE=%.4f" % (pos_x[i], rmses[i]))
            ax.set_ylabel("Depth")
            ax.set_xlabel("Vp")
            ax.invert_yaxis()
        plt.show()

    if obs is not None:
        return wells, true_wells, rmses
    else:
        return wells


def animateinv(it_max, path, project_name, vmin=None, vmax=None, cmap=plt.cm.jet, save=False, save_path="./",
               verbose=0):
    """
    Animates the evolution of an inversion progression outputted by Fullwave3D. Files searched for must be named:
    <project_name>-CPxxxxx-Vp.sgy and be all located in the same path folder.

    Parameters
    ----------
    it_max: int
        max amount of iterations to be animated
    path: str
        path to folder where .sgy model files are located
    project_name: str
        name of the project -- must match exactly
    vmin: float, optional
        min val of color plot scale. See matplotlib.pyplot.contourf parameters. Default None.
    vmax: float, optional
        max val of color plot scale. See matplotlib.pyplot.contourf parameters. Default None.
    cmap: matplotlib.pyplot.cm
        colormap. Default plt.cm.jet
    save: bool, optional
        set to true in order to save the animation. Default False
    save_path: str, optional
        path to save the plot. Default "./"
    verbose: bool, optional
        If set to True will verbose the main steps of the function calculation. Default True.

    Returns
    -------
    None
    """

    # Set verbose
    verbose_print = tools.set_verbose(verbose)

    # Load starting model and get dimensions
    start = tools.load(path + project_name + "-StartVp.sgy", model=True, verbose=0).data

    # Create space to store animation frames
    ani_data = np.zeros((it_max + 1, start.shape[0], start.shape[1]))

    # Add starting model as first animation shot, then load upcoming models
    ani_data[0] = start
    for i in range(1, it_max + 1):
        try:
            ani_data[i] = tools.load(path + project_name + "-CP" + format(i, "05") + "-Vp.sgy",
                                     model=True, verbose=0).data
            verbose_print(str(datetime.datetime.now()) + " \t Loading model %g ...\r" % i)
        except (IndexError, FileNotFoundError):
            sys.stdout.write("\n" + str(datetime.datetime.now()) + " \t Unable to load model %g \r" % i)

    verbose_print("\n" + str(datetime.datetime.now()) + " \t Creating animation...")
    obj = AnimateImshow(ani_data, vmin=vmin, vmax=vmax, cmap=cmap)  # setup the animation
    obj.title = project_name
    obj.xlabel = "x-grid"
    obj.ylabel = "z-grid"
    obj.delay = 100
    obj.run(html=True)  # run it
    if save:
        obj.save_html(save_path + project_name + "-ITER" + format(it_max, "05") + ".html")

    return None
