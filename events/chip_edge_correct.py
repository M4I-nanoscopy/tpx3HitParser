import logging
import numpy as np

logger = logging.getLogger('root')


def chip_edge_correct(e):
    ex = e['x']
    ey = e['y']

    # Take second horizontal row and redivide over itself and two lower pixels
    x_edge1 = np.logical_and(ex > 260, ex < 261)
    ex[x_edge1] = ex[x_edge1] - np.random.random_integers(0, 2, len(ex[x_edge1]))

    # Take first horizontal row and redivide over three pixels
    x_edge2 = np.logical_and(ex > 255, ex < 256)
    ex[x_edge2] = ex[x_edge2] + np.random.random_integers(0, 2, len(ex[x_edge2]))

    # Take second vertical column and redivide over three pixels
    y_edge1 = np.logical_and(ey > 260, ey < 261)
    ey[y_edge1] = ey[y_edge1] - np.random.random_integers(0, 2, len(ey[y_edge1]))

    # Take second vertical column and redivide over three pixels
    y_edge2 = np.logical_and(ey > 255, ey < 256)
    ey[y_edge2] = ey[y_edge2] + np.random.random_integers(0, 2, len(ey[y_edge2]))

    return e