import time


class FPS:
    def __init__(self, avg_window=30):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.frame_count = 0
        self.avg_window = avg_window
        self.times = []

    def update(self):
        now = time.time()
        frame_time = now - self.last_time
        self.last_time = now

        self.times.append(frame_time)
        if len(self.times) > self.avg_window:
            self.times.pop(0)

        self.frame_count += 1

    def instant_fps(self):
        if not self.times:
            return 0.0
        return 1.0 / self.times[-1]

    def average_fps(self):
        if not self.times:
            return 0.0
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time

    def elapsed_time(self):
        return time.time() - self.start_time
