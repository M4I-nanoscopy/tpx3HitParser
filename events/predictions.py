import numpy as np


def calculate_predictions(events, cluster_info):
    # This function takes existing events, and backtracks this to predictions.
    predictions = np.empty((len(cluster_info), 2), dtype=np.float64)

    # The predictions matrices are stored (for historical reasons) in microns. So multiply with pixel size
    # TODO: Make this configurable
    predictions[:, 0] = (events['y'] - cluster_info['y']) * 55000
    predictions[:, 1] = (events['x'] - cluster_info['x']) * 55000

    return predictions
