import numpy as np
from numpy.linalg import linalg

SIZE = 20
MAX_NUMBER = 100
randomMatrix = open('randomMatrix.txt', 'w')
wellConditionedMatrix = open('wellConditionedMatrix.txt', 'w')
poorlyConditionedMatrix = open('poorlyConditionedMatrix.txt', 'w')


def initHilbert(i, j):
    return 1 / (1 + i + j)


def getRandomMatrix():
    # return np.random.rand(SIZE, SIZE)
    return np.random.randint(-MAX_NUMBER, MAX_NUMBER, size=(SIZE, SIZE))


def applyDiagonalDominance(matrix):
    total = 0
    for rows in matrix:
        for elem in rows:
            total += abs(elem)
    total += 1
    for i in range(SIZE):
        matrix[i][i] = total
    return matrix


def printMatrxWithMetaData(file, matrix):
    file.write(np.array2string(matrix) + "\n")
    file.write("determiner:                " + str(linalg.det(matrix)) + "\n")
    file.write("inverse matrix determiner: " + str(linalg.det(linalg.inv(matrix))) + "\n")
    file.write("condition number:          " + str(linalg.cond(matrix)))


printMatrxWithMetaData(wellConditionedMatrix, applyDiagonalDominance(getRandomMatrix()))
printMatrxWithMetaData(randomMatrix, getRandomMatrix())
printMatrxWithMetaData(poorlyConditionedMatrix, np.fromfunction(initHilbert, (SIZE, SIZE)))
