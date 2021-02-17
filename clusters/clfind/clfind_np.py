import numpy as np


def clfind_sub(i, h0, c_id, start_idx, ls, hs, time_window):
    x0, y0, t0 = h0

    # Mark cluster in labels
    ls[i] = c_id

    j = start_idx - 1
    new_start_idx = start_idx
    first_new_start = True

    # Loop over hits, looking for unlabelled hit that is within time window and a neighbour
    for h1 in hs[start_idx:]:
        j = j + 1

        if ls[j] != -1:
            # Put the next start only at the first item not already labeled
            if first_new_start:
                new_start_idx = j
            continue
        else:
            first_new_start = False

        x1, y1, t1 = h1

        if t1 > t0 + time_window:
            break

        if abs(x0 - x1) <= 1 and abs(y0 - y1) <= 1:
            # Recursively find more hits belonging to this same cluster
            new_start_idx = clfind_sub(j, h1, c_id, new_start_idx, ls, hs, time_window)

    return new_start_idx


def clfind(hits, time_window):
    labels = np.full(len(hits), -1)
    cluster_index = 0
    s = 0
    i = 0
    while i < len(hits):
        if labels[i] == -1:
            s = clfind_sub(i, hits[i], cluster_index, s, labels, hits, time_window)

            i = max(s, i)
            cluster_index = cluster_index + 1
        else:
            i = i + 1

    return labels