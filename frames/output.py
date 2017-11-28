import matplotlib.pyplot as plt
import numpy as np
from lib.constants import *
from matplotlib.widgets import Slider


def stats(frame):
    print "Count: %d" % frame.sum()
    print "Mean: %d" % frame.mean()


def show(frame):
    # TODO: Make this cleaner

    # Calculate threshold values
    min5 = np.percentile(frame, 5)
    min = np.min(frame)
    max95 = np.percentile(frame, 95)
    max = np.max(frame)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    im1 = ax.imshow(frame, vmin=min5, vmax=max95)
    fig.colorbar(im1)

    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    smin = Slider(axmin, 'Min', min, max, valinit=min5)
    smax = Slider(axmax, 'Max', min, max, valinit=max95)

    def update(val):
        im1.set_clim([smin.val, smax.val])
        fig.canvas.draw()

    smin.on_changed(update)
    smax.on_changed(update)

    plt.show()


# Display some stats about the SPIDR global timer
def spidr_time_stats(events):
    events = events[()]

    spidr = events[:, SPIDR_TIME]

    tick = 26.843 / 65536.

    print "SPIDR time start: %d" % spidr[0]
    print "SPIDR time end: %d" % spidr[-1]
    print "SPIDR time min: %d" % spidr.min()
    print "SPIDR time max: %d" % spidr.max()

    print "Indeces where SPIDR time jumps: "
    diff = np.where(abs(np.diff(spidr)) > 10000)
    print diff

    print "Seconds (guess):"
    # print tick
    print (spidr[-1] - spidr[0]) * tick + len(diff) * 26.843

    #plot_timers(events)


# Plot SPIDR time of entire run
def plot_timers(events):
    spidr = events[:, SPIDR_TIME]
    plt.scatter(np.arange(len(spidr)), spidr)
    plt.show()
