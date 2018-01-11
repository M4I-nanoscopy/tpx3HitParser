import matplotlib.pyplot as plt
import numpy as np
from lib.constants import *
from matplotlib.widgets import Slider


# Display some stats about the SPIDR global timer
def spidr_time_stats(events):
    # events = events[()]

    spidr = events['TSPIDR']

    tick = 26.843 / 65536.

    print "SPIDR time start: %d" % spidr[0]
    print "SPIDR time end: %d" % spidr[-1]
    print "SPIDR time min: %d" % spidr.min()
    print "SPIDR time max: %d" % spidr.max()

    print "Indeces where SPIDR time jumps: "
    #diff = np.where(abs(np.diff(spidr)) > 1000000)
    #print diff

    print "Seconds (guess):"
    # print tick
    #print (spidr[-1] - spidr[0]) * tick + len(diff) * 26.843
    print (spidr[-1] - spidr[0]) * tick

    plot_timers(events)


# Plot SPIDR time of entire run
def plot_timers(events):
    spidr = events['TSPIDR']
    plt.scatter(np.arange(len(spidr)), spidr)
    plt.show()
