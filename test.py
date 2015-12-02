import aggravate
from aggravate import _aggravate
import numpy as np

from matplotlib import pyplot as plt

Z = np.random.rand(12, 12)

for i in range(_aggravate._n_interpolation):
    output = np.zeros((800, 800))

    aggravate.resample(Z, output, 800/12, 0, 0, 800/12, interpolation=i)

    plt.figimage(output)
    plt.show()
