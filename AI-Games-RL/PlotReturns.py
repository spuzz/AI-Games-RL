import matplotlib.pyplot as plt
import numpy as np


def PlotReturns(returnsSum, title = ""):
    result = np.convolve(returnsSum, np.ones(20) / 20, mode='valid')
    plt.plot(result)
    plt.title(title)
    plt.xlabel("Episode Number")
    plt.ylabel("Moving Average of Returns")
    plt.show()

