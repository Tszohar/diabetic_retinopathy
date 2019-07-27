import time


class Timer:
    def __init__(self, memory_length: int = 100):
        self.memory_length = memory_length

        self._start_time = None
        self.average_time = 0.
        self.calls = 0

    def start(self):
        self.calls += 1
        self._start_time = time.time()

    def stop(self):
        time_diff = time.time() - self._start_time
        memory = self.calls if self.memory_length > self.calls else self.memory_length
        self.average_time -= (self.average_time - time_diff) / memory

    def reset(self):
        self._start_time = None
        self.average_time = 0.
        self.calls = 0
