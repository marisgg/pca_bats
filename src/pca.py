import sklearn.decomposition
import numpy as np

def number_of_components(contribution_rate, eigenvalues):
    total = np.sum(eigenvalues)
    increment = 0.0
    for m, value in enumerate(eigenvalues):
        increment += value
        if increment / total > contribution_rate:
            return m


def custom_pca(matrix):
    if len(matrix.shape) >= 3:
        Exception("Dimensions >=3 not supported in custom_pca.")
    # calculate the mean of each column
    M = np.mean(matrix.T, axis=1)
    # center columns by subtracting column means
    C = matrix - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigendecomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(V)
    m = number_of_components(0.85, eigenvalues)
    # project data
    P = eigenvectors.T.dot(C.T)
    return P.T[:m]

def industry_pca(matrix):
    if len(matrix.shape > 3):
        Exception("Input dimensions >=3 not supported in industry_pca.")
    if len(matrix.shape) == 3:
        samples, nx, ny = matrix.shape
        matrix = matrix.reshape(samples, nx*ny)
    pca = sklearn.decomposition.PCA()
    pca = pca.fit(matrix)
    return pca.transform(matrix)
