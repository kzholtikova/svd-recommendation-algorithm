import numpy as np

REGULARIZATION_CONSTANT = 1e-10

def decompose(A):
    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(A.T, A))  # for right singular vectors V
    eigenvalues = np.abs(eigenvalues)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = eigenvectors[:, idx]  # eigenvectors are the columns of the matrix
    
    singular_values = np.sqrt(eigenvalues)
    S = np.zeros(A.shape)
    np.fill_diagonal(S, singular_values[:min(A.shape)])

    S_inverse = np.diag(1.0 / singular_values)
    U = np.dot(A, np.dot(V, S_inverse))[:, :min(A.shape)]
    U = U / (np.linalg.norm(U, axis=0) + REGULARIZATION_CONSTANT)
    return U, S, V.T


def verify(A, U, S, V_T):
    return np.allclose(A, np.dot(U, np.dot(S, V_T)))
