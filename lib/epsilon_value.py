class EpsilonValue:
    def __init__(self, initial_value=1.0, final_value=0.001, observe_times=5000, explore_times=50000):
        self.__initial_value = initial_value
        self.__final_value = final_value
        self.__observe_times = observe_times
        self.__explore_times = explore_times
        self.reset()

    def reset(self): self.__value = self.__initial_value

    def value(self): self.__value

    def decrement(self, time):
        if self.__is_observation_phase(time) or self.__reach_min_value(): return self.__value

        self.__value -= self.delta()

        return self.value()

    def __delta(self): return (self.__initial_value - self.__final_value) / self.explore

    def __is_observation_phase(self, time): return time <= self.__observe_times

    def __reach_min_value(self): return self.value() < self.__final_value
