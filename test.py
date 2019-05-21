import os

import numpy as np

from conjugate_gradient_method import conjugate_gradient
from generate_matrices import DATA_DIR


def test_method(methodName, fun, A, b):
    x = fun(A=A, b=b)
    print("%s: %s" % (methodName, x))


filenames = ['well_conditioned_matrix', 'random_matrix', 'poorly_conditioned_matrix']
files = [os.path.join(DATA_DIR, name + ".npy") for name in filenames]
b = np.load(os.path.join(DATA_DIR, "b.npy"))
methods = [('conjugate_gradient', conjugate_gradient)]
matrices = [np.load(x) for x in files]
for method in methods:
    for m in matrices:
        n = m.shape[0]
        zero = np.zeros(n)
        test_method(method[0], method[1], m, zero)
