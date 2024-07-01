import numpy as np

REGULARIZATION_CONSTANT = 1e-10  # default close to zero on a logarithmic scale


def svd(A):
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))  # for right singular vectors V
    eigenvalues = np.abs(eigenvalues)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # eigenvectors are the columns of the matrix
    singular_values = np.sqrt(eigenvalues)

    S = np.diag(singular_values)
    V = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    S_regularized = np.diag(np.maximum(REGULARIZATION_CONSTANT, singular_values))  # handles singular matrices
    U = np.dot(A, np.dot(V, np.linalg.inv(S_regularized)))
    U = U / np.linalg.norm(U, axis=0)
    return U, S, V.T


def verify_svd(A, U, S, V_T):
    return np.allclose(A, np.dot(U, np.dot(S, V_T)))
