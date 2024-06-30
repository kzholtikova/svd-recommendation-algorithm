import numpy as np


def svd(A):
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))
    idx = np.argsort(eigenvalues[::-1])
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # eigenvectors are the columns of the matrix

    print("Eigenvalues:", eigenvalues)
    S = np.diag(np.sqrt(eigenvalues))
    V = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    U = np.dot(A, np.dot(V, np.linalg.inv(S)))
    U = U / np.linalg.norm(U, axis=0)
    return U, S, V.T


def verify_svd(A, U, S, V_T):
    return np.allclose(A, np.dot(U, np.dot(S, V_T)))


A = np.random.rand(2, 3)
U, S, V_T = svd(A)
print("Verification:", verify_svd(A, U, S, V_T))
print("A:", A)
print("U*S*V_T:", np.dot(U, np.dot(S, V_T)))
