import numpy as np


def conjugate_gradient(A, b, x=None, accuracy=1e-5):
    """
    Solve Ax = b with conjugate gradient method
    :param A: 2d array
    :param b: 1d array
    :param x: 1d array, initial point
    :param accuracy: answer accuracy
    :return: 1d array of x, such that Ax = b
    """
    n = len(b)
    if not x:
        x = np.ones(n)
    r = b - np.dot(A, x)
    z = r
    r2 = np.dot(r, r)
    for i in range(2 * n):
        Az = np.dot(A, z)
        alpha = r2 / np.dot(Az, z)
        x += alpha * z
        r -= alpha * Az
        r2_new = np.dot(r, r)
        beta = r2_new / r2
        r2 = r2_new
        if r2_new < accuracy:
            break
        z = r + beta * z
    return x
