import math


def jacobi(A, b, accuracy=1e-5, iterations=10000):
    iteration = 0
    B = []
    d = []
    xn = []
    x = []

    for i in range(0, len(A)):
        d.append(b[i] / A[i][i])
        xn.append(d[i])
        x.append(d[i] + 1000)
        B.append([])
        for j in range(0, len(A[i])):
            B[i].append(-A[i][j] / A[i][i])
        B[i][i] = 0.0

    # print("NORM: " + str(linalg.norm(B, 'fro')))
    # print("DET: " + str(linalg.det(B)))

    while (delta_vectors_norm(x, xn) > accuracy) and iteration < iterations:
        x = xn.copy()
        for i in range(0, len(B)):
            xn[i] = 0.0
            for j in range(0, len(B[i])):
                xn[i] += B[i][j] * x[j]
            xn[i] += d[i]
        iteration = iteration + 1

    print("Iterations number: " + str(iteration))
    return x


def delta_vectors_norm(x, y):
    res = 0.0
    for i in range(0, len(x)):
        res += (y[i] - x[i]) ** 2
    return math.sqrt(res)
