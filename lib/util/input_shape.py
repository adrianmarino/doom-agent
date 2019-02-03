class InputShape:
    def __init__(self, rows, cols, channels):
        self.rows = rows
        self.cols = cols
        self.channels = channels

    def as_tuple(self): return self.rows, self.cols, self.channels
