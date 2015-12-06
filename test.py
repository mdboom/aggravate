import aggravate
from aggravate import _aggravate
import numpy as np

from matplotlib import pyplot as plt

Z = np.random.rand(12, 12)

mtx = np.zeros((3, 3))
mtx[0, 0] = 800/12
mtx[1, 1] = 800/12
mtx[2, 2] = 1

for i in range(_aggravate._n_interpolation):
    output = np.zeros((800, 800))

    aggravate.resample(Z, output, mtx, interpolation=i)

    plt.figimage(output)
    plt.show()
