import os

import numpy as np
from numpy.linalg import linalg

SIZE = 20
MAX_NUMBER = 100
DATA_DIR = "data"


def init_hilbert(i, j):
    return 1 / (1 + i + j)


def get_random_matrix():
    return np.random.randint(-MAX_NUMBER, MAX_NUMBER, size=(SIZE, SIZE))


def get_random_vector():
    return np.random.randint(-MAX_NUMBER, MAX_NUMBER, size=(SIZE,))


def apply_diagonal_dominance(matrix):
    total = 0
    for rows in matrix:
        for elem in rows:
            total += abs(elem)
    total += 1
    for i in range(SIZE):
        matrix[i][i] = total
    return matrix


def print_matrix_with_meta_data(filename, matrix):
    np.save(os.path.join(DATA_DIR, filename), matrix)
    details_file = open(os.path.join(DATA_DIR, filename + '.txt'), 'w')
    try:
        details_file.write(np.array2string(matrix) + "\n")
        details_file.write("\ndeterminer:                " + str(linalg.det(matrix)) + "\n")
        details_file.write("inverse matrix determiner: " + str(linalg.det(linalg.inv(matrix))) + "\n")
        details_file.write("euclidean norm:            " + str(linalg.norm(matrix, 'fro')) + "\n")
        details_file.write("euclidean norm \nof inverse matrix:         " + str(linalg.norm(linalg.inv(matrix), 'fro'))
                           + "\n")
        details_file.write("condition number:          " + str(linalg.cond(matrix, 'fro')))
    finally:
        details_file.close()


def print_b(filename, matrix):
    np.save(os.path.join(DATA_DIR, filename), matrix)
    details_file = open(os.path.join(DATA_DIR, filename + '.txt'), 'w')
    try:
        details_file.write(np.array2string(matrix))
    finally:
        details_file.close()


if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

print_b('b', get_random_vector())
print_matrix_with_meta_data('well_conditioned_matrix', apply_diagonal_dominance(get_random_matrix()))
print_matrix_with_meta_data('random_matrix', get_random_matrix())
print_matrix_with_meta_data('poorly_conditioned_matrix', np.fromfunction(init_hilbert, (SIZE, SIZE)))
