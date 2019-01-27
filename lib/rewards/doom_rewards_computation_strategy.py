class DoomRewardsComputationStrategy:
    def calculate(self, env, rewards, current_state, next_state):
        if self.__kill_count(current_state) > self.__kill_count(next_state):
            rewards += 1
        if self.__ammo_count(current_state) < self.__ammo_count(next_state):
            rewards -= 1
        if self.__health_count(current_state) < self.__health_count(next_state):
            rewards -= 1
        return rewards

    @staticmethod
    def __kill_count(state): return state.variables()[0]

    @staticmethod
    def __ammo_count(state): return state.variables()[1]

    @staticmethod
    def __health_count(state): return state.variables()[2]
