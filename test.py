import os

import numpy as np
from numpy.linalg import LinAlgError

from conjugate_gradient_method import conjugate_gradient

DATA_DIR = "data"
ACCURACY = 1e-5


def test_method(method_name, matrix_name, solution, fun, A, b):
    x = fun(A=A, b=b)
    is_equal = np.allclose(solution, x, 0, ACCURACY, False)
    print('%s (%s): %s' % (method_name, matrix_name, "EQUAL" if is_equal else "NOT EQUAL"))
    print(str(x))


filename_arr = ['well_conditioned_matrix', 'random_matrix', 'poorly_conditioned_matrix']
b = np.load(os.path.join(DATA_DIR, "b.npy"))
methods = [('conjugate_gradient', conjugate_gradient)]
matrices = [(name, np.load(os.path.join(DATA_DIR, name + ".npy"))) for name in filename_arr]
solutions = [np.linalg.solve(m[1], b) for m in matrices]
for matrix in matrices:
    try:
        print("solution (%s) :" % str(matrix[0]))
        print(np.linalg.solve(matrix[1], b))
    except LinAlgError as e:
        print("error: A is singular")
for method in methods:
    try:
        for (solution, matrix) in zip(solutions, matrices):
            test_method(method[0], matrix[0], solution, method[1], matrix[1], b)
    except Exception as e:
        print("%s (%s): %s" % ("exception", method[0], str(e)))
