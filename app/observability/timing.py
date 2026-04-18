import time


class Timer:
    def __init__(self) -> None:
        self._start: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        return (time.perf_counter() - self._start) * 1000
