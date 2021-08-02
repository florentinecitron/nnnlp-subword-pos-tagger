from collections import deque
from dataclasses import dataclass

@dataclass
class RollingAverage:
    # for keeping track of loss
    n: int
    total: int
    q: deque

    @classmethod
    def make(cls, n):
        return cls(n, 0, deque())

    def add_stat(self, stat):
        if len(self.q) < self.n:
            self.total += stat
            self.q.append(stat)
        else:
            to_remove = self.q.popleft()
            self.total -= to_remove
            self.total += stat
            self.q.append(stat)

    def get_val(self):
        return self.total / len(self.q)
