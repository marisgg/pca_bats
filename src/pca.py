import sklearn.decomposition
import numpy as np

class PCA():
    def custom_pca(self, matrix, numOfComponents=2):
        if len(matrix.shape) >= 3:
            Exception("Dimensions >=3 not supported in custom_pca.")
        # calculate the mean of each column
        M = np.mean(matrix.T, axis=1)
        # center columns by subtracting column means
        C = matrix - M
        # calculate covariance matrix of centered matrix
        V = np.cov(C.T)
        # eigendecomposition of covariance matrix
        _, vectors = np.linalg.eig(V)
        # project data
        P = vectors.T.dot(C.T)
        return P.T

    def industry_pca(self, matrix, numOfComponents=2):
        if len(matrix,shape) > 3:
            Exception("Dimensions > 3 not (yet) supported in industry_pca.")
        if len(matrix.shape) == 3:
            samples, nx, ny = matrix.shape
            matrix = matrix.reshape(samples, nx*ny)
            numOfComponents = 3
        pca = sklearn.decomposition.PCA(numOfComponents)
        pca = pca.fit(matrix)
        return pca.transform(matrix)