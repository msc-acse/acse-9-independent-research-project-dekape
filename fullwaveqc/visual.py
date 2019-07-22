import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as an
from IPython.core.display import display, HTML
import matplotlib
from scipy.stats import spearmanr
import copy
import fullwaveqc.tools as tools

# Setting matplotlib plotting parameters
plt.rcParams.update({'font.size': 14})
matplotlib.rcParams['animation.embed_limit'] = 2**128


class AnimateImshow():
    def __init__(self, arrays, dt=1):
        """
        Array is a list of 2d arrays or a 3d array with time on axis=0.
        Note in jupyter the animation needs to be done"
        """
        # setup data
        self.arr = arrays
        self.Nt = len(arrays)
        self.dt = dt

        # setup the figure
        self.fig = plt.figure(figsize=(20, 20))
        self.im = plt.imshow(arrays[0], animated=True, cmap=plt.cm.jet, vmin=1400, vmax=4000)
        self.ax = plt.gca()
        self.ax.invert_yaxis()
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
                                    interval=self.delay, blit=True);

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


def amplitude(SegyData, shot=1, cap=0., levels=100, vmin=None, vmax=None, save=False, cmap=plt.cm.seismic,
        xstart=0, xend=None, wstart=0, wend=None, save_path="./FIGURES/"):

    # Get data from SegyData object
    dt = SegyData.dt[shot-1]
    wstart = int(wstart/dt)
    if wend is not None:
        wend = int(wend/dt)
    data = SegyData.data[shot-1][xstart:xend, wstart:wend]

    # Cap amplitude values
    data[np.where(np.logical_and(data >= (-1) * cap, data <= cap))] = 0.

    # Plot figure
    figure, ax = plt.subplots(1, 1)
    figure.set_size_inches(18.5, 10.5)

    # Plotting and adjusting axis
    if xend is None:
        xend = xend = data.shape[0]
    x = np.arange(xstart, xend, 1)

    if wend is None:
        wend = data.shape[1]
    y = np.linspace(wend, wstart, data.shape[1])  * dt

    MP = ax.contourf(x, y, np.rot90(data),
                     cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
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

    return None


def interamp(SegyData1, SegyData2, shot=1, shot2=None, n_blocks=2, cap=0., levels=100, vmin=None, vmax=None,
             cmap=plt.cm.seismic, wstart=0., wend=None, save=False, save_path="./FIGURES/" ):

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

    InterData = copy.copy(SegyData1)
    InterData.data[shot-1] = data_inter
    InterData.name = SegyData1.name + "-INTERAMP"
    amplitude(InterData, shot=shot, cap=cap, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap, save=save,
              wstart=wstart, wend=wend, save_path=save_path)
    return None


def wiggle(SegyData, shot=1, scale=5, skip_trace=0, skip_time=0, wstart=0., wend=None, xstart=0, xend=None,
           delay_samples=0, save=False, save_path="./FIGURES/"):

    # Get data from SegyData object
    dt = SegyData.dt[shot-1]
    data = SegyData.data[shot-1]

    # Stack delay to data
    delay = np.zeros((data.shape[0], delay_samples))
    data = np.hstack((delay, data))

    offsets = np.arange(0, data.shape[0], skip_trace+1)
    times = np.arange(0, data.shape[1], skip_time + 1)  * SegyData.dt[shot-1]

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


def interwiggle(SegyData1, SegyData2, shot=1, shot2=None, overlay=False, scale=5, skip_trace=20, skip_time=0, delay_samples=0,
                wstart=0., wend=None, xstart=0, xend=None, label1="Observed", label2="Predicted", save=False, save_path="./FIGURES/"):

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

            x1 = scale * data1[i] + i
            ax.plot(x1, times, '-k')
            ax.fill_betweenx(times, i, x1, where=(x1 > i), color='k')

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
    ax.set_title(SegyData1.name + "-INTERWIGGLE")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Rec x")
    ax.set_ylim(wend, wstart)
    ax.set_xlim(xstart, xend)
    ax.legend(custom_lines, [label1, label2], loc=7, bbox_to_anchor=(1.1, 0.5))

    if save:
        plt.savefig(save_path + SegyData1.name + "-INTERWIGGLE"+".png", dpi=300)
    else:
        plt.show()

    return None


def vpmodel(Model, cap=0., levels=200, vmin=1450, vmax=3200, cmap=plt.cm.jet, units="m/s",
            save=False, save_path="./FIGURES/"):

    # Get data from Model
    data = Model.data
    dx = Model.dx
    name = Model.name

    # Cap values
    data[np.where(np.logical_and(data >= (-1)*cap, data <= cap))] = 0.

    # Plot figure
    figure, ax = plt.subplots(1, 1)
    figure.set_size_inches(18.5, 10.5)

    # Contour plot and formatting axis
    x = np.arange(0, data.shape[1], 1)*dx
    y = np.arange(data.shape[0], 0, -1)*dx
    MP = ax.contourf(x, y, data,
                     cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_title(name)
    ax.set_ylabel("Depth")
    ax.set_xlabel("X")


    # Format limits of colorbar
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(data)
    m.set_clim(vmin, vmax)
    # m.set_label(units)
    cbar = plt.colorbar(m)
    cbar.set_label(units)

    if save:
        plt.savefig(save_path + name + ".png", dpi=300)
    else:
        plt.show()

    return None


def vpwell(Model, pos_x, TrueModel=None,  plot=True):

    # Get model data
    pred = np.rot90(np.rot90(Model.data.T))

    # Get true model
    if TrueModel is not None:
        obs = np.rot90(np.rot90(TrueModel.data.T))
    else:
        obs = None

    n = len(pos_x)
    wells = []
    true_wells = []
    rmses = []


    figure, axs = plt.subplots(1, n)
    figure.set_size_inches(9*n, 20)
    for i, ax in enumerate(axs):
        # Get well data for predicted model
        grid_pos = int(pos_x[i]/Model.dx)
        well = pred[grid_pos, :]
        wells.append(well)

        # Get Well data for observed model
        if obs is not None:
            true_grid_pos = int(pos_x[i]/TrueModel.dx)
            true_well = obs[true_grid_pos, :]
            true_wells.append(true_well)

        # If not axis given for plot, create figure
        if (ax is None) and plot:
            figure, ax = plt.subplots(1, 1)
            figure.set_size_inches(8, 20)

        ypred = np.arange(0, well.shape[0], 1) * Model.dx

        # Calculate RMSE, interpolate predicted data first
        if obs is not None:
            ytrue = np.arange(0, true_well.shape[0], 1) * TrueModel.dx
            well_interp = np.interp(ytrue, ypred, well)
            rho, p = spearmanr(true_well, well_interp)
            rmse = np.sqrt(np.mean((well_interp - true_well)**2))
            rmses.append(rmse)
        else:
            rmse=0.0

        # Plot axis
        if plot:
            if obs is not None:
                ax.plot(true_well, ytrue, 'k', label="Observed")
            ax.plot(well, ypred , 'r', label="Predicted")
            ax.set_title(Model.name+r" x=%i, RMSE=%.4f" % (pos_x[i], rmse))
            ax.set_ylabel("Depth")
            ax.set_xlabel("Vp")
            ax.legend(loc="best")
            ax.invert_yaxis()

    if plot:
        plt.show()

    if obs is not None:
        return wells, true_wells, rmses
    else:
        return wells



def animateinv(it, path, project_name):
    # Load starting model and get dimensions
    start = tools.load(path + project_name + "/" + project_name + "-StartVp.sgy", model=True)[0][
        0]  # [:, 0, :]

    # [0] instead of [:, 0, :] for true and starting models
    ani_data = np.zeros((it + 1, start.shape[1], start.shape[0]))

    # Add starting model as first animation shot, then load upcoming models
    ani_data[0] = np.rot90(start)
    for i in range(1, it + 1):
        ani_data[i] = np.rot90(tools.load("/geophysics2/dpelacani/PROJECTS/" + project_name + "/" + project_name + \
                                         "-CP" + format(i, "05") + "-Vp.sgy", verbose=0)[0][:, 0, :])
    obj = AnimateImshow(ani_data)  # setup the animation
    obj.title = project_name
    obj.xlabel = "x-grid"
    obj.ylabel = "z-grid"
    obj.delay = 100
    obj.run(html=True)  # run it
    obj.save_html("./FIGURES/" + project_name + "-ITER" + format(it, "05") + ".html")

    return None









