import numpy as np
import scipy.stats as stats

class RollingBuffer:

    def __init__(self, buffer_len):
        self.__buffer = np.zeros(buffer_len)
        self.__counter = 0
        self.__buffer_len = buffer_len

    def append(self, data):
        self.__buffer = np.roll(self.__buffer, -1)
        self.__buffer[-1] = data
        self.__counter += 1
        if self.__counter > self.__buffer_len:
            self.__counter = self.__buffer_len

    def values(self):
        return self.__buffer[-self.__counter:]

    def mean(self):
        return np.sum(self.__buffer) / self.__counter

    def indices(self):
        return range(self.__counter)

    def clear(self):
        self.__counter = 0

    def length(self):
        return self.__buffer_len

    def full(self):
        return self.__counter == self.__buffer_len

    def slope_probability_less_than(self, prob):
        idxs = self.indices()
        n = len(idxs)
        if n < 3:
            return 1
        values = self.values()
        n = float(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(idxs, values)
        residuals = idxs * slope + intercept
        variance = np.sum(np.power(residuals - values, 2)) / (n - 2)
        slope_std_error = np.sqrt(variance * (12.0 / (np.power(n, 3) - n)))
        p_less_than_zero = stats.norm.cdf(prob, slope, slope_std_error)
        return p_less_than_zero
