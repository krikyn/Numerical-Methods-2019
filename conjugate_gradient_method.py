import numpy as np


def conjugate_gradient(A, b, x0=None, accuracy=1e-5, iterations=10000):
    """
    Solve Ax = b with conjugate gradient method
    :param A: 2d array
    :param b: 1d array
    :param x0: 1d array, initial point
    :param accuracy: answer accuracy
    :param iterations: maximum number of iterations
    :return: 1d array of x, such that Ax = b
    """
    n = len(b)
    norm_b = np.linalg.norm(b)
    AT = np.transpose(A)
    if not x0:
        x0 = np.ones(n)
    r = b - np.dot(A, x0)
    p = r.copy()
    z = r.copy()
    s = r.copy()
    pr = np.dot(p, r)
    for i in range(0, iterations):
        Az = np.dot(A, z)
        alpha = pr / np.dot(s, Az)
        x0 += np.dot(alpha, z)
        r -= np.dot(alpha, Az)
        p -= np.dot(alpha, np.dot(AT, s))
        pr_new = np.dot(p, r)
        beta = pr_new / pr
        pr = pr_new
        if np.linalg.norm(r) / norm_b < accuracy:
            break
        z = r + np.dot(beta, z)
        s = p + np.dot(beta, s)
    return x0
