import numpy as np


def gauss_method(A, b):
    n = len(A)
    b = np.reshape(b, (-1, 1))
    M = np.append(A, b, axis=1)

    n = len(M)

    for i in range(0, n):
        maxEl = abs(M[i][i])
        maxRow = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > maxEl:
                maxEl = abs(M[k][i])
                maxRow = k

        for k in range(i, n + 1):
            tmp = M[maxRow][k]
            M[maxRow][k] = M[i][k]
            M[i][k] = tmp

        for k in range(i + 1, n):
            c = -M[k][i] / M[i][i]
            for j in range(i, n + 1):
                if i == j:
                    M[k][j] = 0
                else:
                    M[k][j] += c * M[i][j]

    x = [0 for i in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n] / M[i][i]
        for k in range(i - 1, -1, -1):
            M[k][n] -= M[k][i] * x[i]

    return x
