class RollingMeanVar:
    def __init__(self, window_size = 10):
        self.window_size = 10
        self.num_items = 0
        self.idx = 0
        self.data = [0.0] * self.window_size
        self.warmed_up = False

    def add_datum(self, datum):
        self.data[self.idx] = datum
        self.idx = (self.idx + 1) % self.window_size
        self.num_items += 1
        self.warmed_up = self.num_items >= self.window_size

        self.mean = sum(self.data) / self.window_size
        self.var = sum((d - self.mean)**2 for d in self.data) / self.window_size

        
