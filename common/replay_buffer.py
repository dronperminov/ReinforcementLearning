import numpy as np


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def clear(self):
        self.buffer.clear()
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        replay = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        if len(self.buffer) < self.max_size:
            self.buffer.append(replay)
        else:
            self.buffer[self.position] = replay
            self.position = (self.position + 1) % self.max_size

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        return np.random.choice(self.buffer, batch_size)
