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

    if spidr.max() > 65535 - 100:
        ticks = 65535 - int(spidr[0]) + int(spidr[-1])
    else:
        ticks = int(spidr[-1]) - int(spidr[0])

    print("Seconds exposure time (estimate): %.5f" % (ticks * tick))

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
