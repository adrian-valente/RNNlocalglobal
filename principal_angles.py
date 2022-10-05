import numpy as np
import scipy.linalg


def principal_angles(U, V, radians=False):
    """
    Computes the sequence of principal angles between two linear subspaces U and V.
    https://en.wikipedia.org/wiki/Angles_between_flats

    :param U: np array of shape (nvecs, d) or simply (d) representing a basis of a subspace spanned by nvecs vectors
    :param V: same
    :param radians: bool, if False output in degrees
    :return: angles, ndarray of shape (nangles) where nangles is the minimal subspace dimension between that of U and V
    """
    # input management
    if len(U.shape) == 1:
        U = U[np.newaxis, :]
    if len(V.shape) == 1:
        V = V[np.newaxis, :]

    # compute principal angles
    Uo = scipy.linalg.orth(U.T)
    Vo = scipy.linalg.orth(V.T)
    G = Uo.T @ Vo
    _, s, _ = scipy.linalg.svd(G)
    s = np.clip(s, 0, 1)  # Avoid bad conditioning
    if not radians:
        return np.arccos(s) * 180 / np.pi
    if radians:
        return np.arccos(s)


if __name__ == '__main__':
    # Test 1
    U = np.array([1, 1, 1])
    V = np.array([2, 2, 2])
    print(principal_angles(U, V))  # should be 0

    # Test 2
    U = np.array([1, 1, 1])
    V = np.array([-2, -2, -2])
    print(principal_angles(U, V))  # should be 0

    # Test 3
    U = np.array([1, 0, 0])
    V = np.array([[0, 1, 0], [0, 0, 1]])
    print(principal_angles(U, V))  # should be 90 degrees

    # Test 4
    U = np.array([[1, 0, 0], [0, 1, 0]])
    V = np.array([[0, 1, 0], [0, 0, 1]])
    print(principal_angles(U, V))  # should be 0 and 90 degrees