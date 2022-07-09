import numpy as np


class UniformSpace:
    def __init__(self, low: float, high: float, count: int):
        self.low = low
        self.high = high
        self.shape = count

    def sample(self):
        return np.random.uniform(self.low, self.high, [self.shape])
