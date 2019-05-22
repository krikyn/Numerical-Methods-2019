import math


def seidel_relax(A, b, accuracy=1e-5, iterations=1000):
    iteration = 0
    B = []
    d = []
    xn = []
    x = []
    relax = 0.5

    for i in range(0, len(A)):
        d.append(b[i] / A[i][i])
        xn.append(d[i])
        x.append(d[i] + 1000)
        B.append([])
        for j in range(0, len(A[i])):
            B[i].append(-A[i][j] / A[i][i])
        B[i][i] = 0.0

    while (delta_vectors_norm(x, xn) > accuracy) and iteration < iterations:
        x = xn.copy()
        for i in range(0, len(B)):
            xn[i] = 0.0
            for j in range(0, len(B[i])):
                xn[i] += B[i][j] * (xn[j] if (j < i) else x[j])
            xn[i] += d[i]
            xn[i] = relax * xn[i] + (1 - relax) * x[i]
        iteration = iteration + 1

    return x


def delta_vectors_norm(x, y):
    res = 0.0
    for i in range(0, len(x)):
        res += (y[i] - x[i]) ** 2
    return math.sqrt(res)
