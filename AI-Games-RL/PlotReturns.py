import matplotlib.pyplot as plt
import numpy as np


def PlotReturns(returnsSum):
    result = np.convolve(returnsSum, np.ones(20) / 20, mode='valid')
    plt.plot(result)
    plt.show()
