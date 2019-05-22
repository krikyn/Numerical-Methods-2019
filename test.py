import os
import traceback

import numpy as np
from numpy.linalg import LinAlgError

from conjugate_gradient_method import conjugate_gradient
from gauss_method import gauss_method
from jacobi_method import jacobi
from seidel_method import seidel
from seidel_relax_method import seidel_relax

DATA_DIR = "data"
ACCURACY = 1e-5


def test_method(method_name, matrix_name, solution, fun, A, b):
    x = fun(A=A, b=b)
    is_equal = np.allclose(solution, x, 0, ACCURACY, False)
    print('%s (%s): %s' % (method_name, matrix_name, "EQUAL" if is_equal else "NOT EQUAL"))
    print(str(x))


np.warnings.filterwarnings('ignore')
filename_arr = ['well_conditioned_matrix', 'random_matrix', 'poorly_conditioned_matrix']
b = np.load(os.path.join(DATA_DIR, "b.npy"))
methods = [('conjugate_gradient', conjugate_gradient), ('gauss', gauss_method), ('jacobi', jacobi), ('seidel', seidel),
           ('seidel_relax', seidel_relax)]
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
            test_method(method[0], matrix[0], solution, method[1], matrix[1].copy(), b.copy())
    except Exception as e:
        print("%s (%s): %s" % ("exception", method[0], e))
        traceback.print_exc()
