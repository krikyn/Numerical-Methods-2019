import numpy as np


def gauss_method(A, b):
    n = len(A)
    b = np.reshape(b, (-1, 1))
    M = np.append(A, b, axis=1)

    for k in range(n):
        for i in range(k, n):
            if abs(M[i][k]) > abs(M[k][k]):
                M[k], M[i] = M[i], M[k]
            else:
                pass

        for j in range(k + 1, n):
            q = M[j][k] / M[k][k]
            for m in range(k, n + 1):
                M[j][m] -= q * M[k][m]

    x = np.zeros(n)

    x[n - 1] = M[n - 1][n] / M[n - 1][n - 1]
    for i in range(n - 1, -1, -1):
        z = 0
        for j in range(i + 1, n):
            z = z + M[i][j] * x[j]
        x[i] = (M[i][n] - z) / M[i][i]
    return x
