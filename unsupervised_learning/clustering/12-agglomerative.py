"""Task 12: Agglomerative"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset with Ward linkage and
        displays the dendrogram
    Args:
        X: np.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters
    Returns: clss"""

    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion='distance')
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    return clss
