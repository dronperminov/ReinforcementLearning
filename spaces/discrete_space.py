import numpy as np


class DiscreteSpace:
    def __init__(self, count: int, start: int = 0):
        self.start = start
        self.shape = count

    def sample(self):
        return self.start + np.random.randint(self.shape)

    def probabilities_sample(self, probs: np.ndarray):
        return self.start + np.random.choice(self.shape, p=probs)

    def contains(self, action: int):
        return 0 <= action - self.start < self.shape
