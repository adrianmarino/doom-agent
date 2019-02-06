class EpsilonValue:
    def __init__(self, initial_value, final_value, explore_times):
        self.__initial_value = initial_value
        self.__final_value = final_value
        self.__delta = (initial_value - final_value) / explore_times
        self.reset()

    def decrement(self):
        if self.__value > self.__final_value:
            self.__value -= self.__delta
        return self.__value

    def reset(self):
        self.__value = self.__initial_value

    def value(self):
        return self.__value
