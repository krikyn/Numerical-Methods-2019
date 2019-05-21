import os

import numpy as np

from conjugate_gradient_method import conjugate_gradient

DATA_DIR = "data"


def test_method(method_name, matrix_name, fun, A, b):
    x = fun(A=A, b=b)
    print('%s (%s)' % (method_name, matrix_name))
    print(str(x))


filename_arr = ['well_conditioned_matrix', 'random_matrix', 'poorly_conditioned_matrix']
b = np.load(os.path.join(DATA_DIR, "b.npy"))
methods = [('conjugate_gradient', conjugate_gradient)]
matrices = [(name, np.load(os.path.join(DATA_DIR, name + ".npy"))) for name in filename_arr]
for method in methods:
    try:
        for matrix in matrices:
            test_method(method[0], matrix[0], method[1], matrix[1], b)
    except Exception as e:
        print("%s (%s): %s" % ("exception", method[0], str(e)))
