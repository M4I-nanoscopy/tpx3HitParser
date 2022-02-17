import logging
import numpy as np

logger = logging.getLogger('root')


def chip_edge_correct(e):
    ex = e['x']
    ey = e['y']

    # Shift x and y coordinate by 4 pixels for events right/bottom of the edge
    ex[ex > 257] = ex[ex > 257] + 4
    ey[ey > 257] = ey[ey > 257] + 4

    # Take second horizontal row and redivide over itself and two lower pixels
    x_edge1 = np.logical_and(ex > 256, ex < 257)
    ex[x_edge1] = ex[x_edge1] + np.random.random_integers(2, 4, len(ex[x_edge1]))

    # Take first horizontal row and redivide over three pixels
    x_edge2 = np.logical_and(ex > 255, ex < 256)
    ex[x_edge2] = ex[x_edge2] + np.random.random_integers(0, 2, len(ex[x_edge2]))

    # Take second vertical column and redivide over three pixels
    y_edge1 = np.logical_and(ey > 256, ey < 257)
    ey[y_edge1] = ey[y_edge1] + np.random.random_integers(2, 4, len(ey[y_edge1]))

    # Take second vertical column and redivide over three pixels
    y_edge2 = np.logical_and(ey > 255, ey < 256)
    ey[y_edge2] = ey[y_edge2] + np.random.random_integers(0, 2, len(ey[y_edge2]))

    return e