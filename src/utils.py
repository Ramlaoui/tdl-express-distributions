import numpy as np
import ot


def compute_wasserstein_distance(y, y_pred, bins=100):
    # We compute the empirical distribution of Y and Y_pred
    hist, bins_edges = np.histogram(y, bins=bins, density=True)
    p = hist * np.diff(bins_edges)

    hist_pred, bins_edges_pred = np.histogram(
        y_pred.detach().numpy(), bins=bins, density=True
    )
    q = hist_pred * np.diff(bins_edges_pred)

    # We compute the cost matrix
    M = ot.dist(bins_edges.reshape(-1, 1)[:-1], bins_edges_pred.reshape(-1, 1)[:-1])

    # We compute the Wasserstein distance
    W2 = ot.emd2(p, q, M)

    return W2
