class AVG:
    def __init__(self): self.reset()

    def reset(self):
        self.__count = 0
        self.__values_sum = 0

    def next(self, value):
        self.__count += 1
        self.__values_sum += value
        return self.__count

    def name(self): return 'avg'

    def result(self): return self.__values_sum / self.__count
