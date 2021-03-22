import sklearn.decomposition
import numpy as np

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
    eivenvalues, eigenvectors = np.linalg.eig(V)
    # project data
    P = eigenvectors.T.dot(C.T)
    return P.T

def industry_pca(matrix):
    if len(matrix.shape > 3):
        Exception("Input dimensions >=3 not supported in industry_pca.")
    if len(matrix.shape) == 3:
        samples, nx, ny = matrix.shape
        matrix = matrix.reshape(samples, nx*ny)
    pca = sklearn.decomposition.PCA()
    pca = pca.fit(matrix)
    return pca.transform(matrix)