class InputShape:
    @staticmethod
    def from_str(value):
        shape = eval(value)
        return InputShape(shape[0], shape[1], shape[2])

    def __init__(self, rows, cols, channels):
        self.rows = rows
        self.cols = cols
        self.channels = channels

    def as_tuple(self): return self.rows, self.cols, self.channels
