import numpy as np

REGULARIZATION_CONSTANT = 1e-10

def decompose(A):
    U, eigenvalues = findSingularVectors(A)
    V = findSingularVectors(A.T)[0]
    
    singular_values = np.sqrt(eigenvalues)
    S = np.zeros(A.shape)
    np.fill_diagonal(S, singular_values[:A.shape[0]])
        
    if A.shape[0] > A.shape[1]:
        V = coordinate_vectors(V, A.T, U, singular_values)
    else:
        U = coordinate_vectors(U, A, V, singular_values)
        
    return U, S, V.T


def findSingularVectors(A):
    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(A, A.T)) 
    eigenvalues = np.abs(eigenvalues)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    return eigenvectors[:, idx], eigenvalues


def coordinate_vectors(L, A, R, singular_values):
    for i in range(L.shape[0]):
        L[:, i] = np.dot(A, R[:, i]) * (1 / singular_values[i])
    return L


def verify(A, U, S, V_T):
    return np.allclose(A, np.dot(U, np.dot(S, V_T)))
