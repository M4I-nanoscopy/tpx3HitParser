import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter


# Display some stats about the SPIDR global time
def spidr_time_stats(events):
    spidr = events['TSPIDR']

    tick = 26.843 / 65536.

    print("SPIDR time start: %d" % spidr[0])
    print("SPIDR time end: %d" % spidr[-1])
    print("SPIDR time min: %d" % spidr.min())
    print("SPIDR time max: %d" % spidr.max())

    # print "Indices where SPIDR time jumps: "
    # diff = np.where(abs(np.diff(spidr)) > 1000000)
    # print diff

    print("Seconds exposure time (guess):")
    # print (spidr[-1] - spidr[0]) * tick + len(diff) * 26.843
    print(float(spidr[-1] - spidr[0]) * tick)

    plot_timers(events)


# Plot SPIDR time of entire run
def plot_timers(events):
    fig, ax = plt.subplots()

    index = np.arange(len(events))

    for chip in range(0, 4):
        # Index frame to only the particular chip
        chip_events = events[[events['chipId'] == chip]]
        chip_index = index[[events['chipId'] == chip]]

        # Get only every 1000nth hit
        spidr = chip_events['TSPIDR'][1::1000]
        spidr_index = chip_index[1::1000]

        plt.scatter(spidr_index, spidr, label='Chip %d' % chip)

    plt.title('SPIDR time (every 1000nth hit)')

    formatter0 = EngFormatter(unit='hit')
    ax.xaxis.set_major_formatter(formatter0)

    plt.xlabel('Hit index')
    plt.ylabel('SPIDR time ticks')
    plt.legend()
    plt.show()
