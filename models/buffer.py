from collections import deque
from random import sample
from typing import Any
from random import sample


class ReplayBuffer:
    def __init__(self, max_len: int):
        self.buffer: deque[Any] = deque([], maxlen=max_len)

    def add(self, data: Any):
        self.buffer.append(data)
        return

    def sample(self, batchSize: int):
        # returns a list with the batches as a list at each index
        x = sample(self.buffer, batchSize)
        return list(zip(*x))

    def alt_sample(self, batchSize: int):
        # might work, might not
        indices = sample(range(len(self.buffer)), batchSize)
        batch = [
            [self.buffer[x][y] for x in indices] for y in range(len(self.buffer[0]))
        ]
        return batch
