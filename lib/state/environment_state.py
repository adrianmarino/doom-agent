class EnvironmentState:
    def __init__(self, state, variable_names):
        self.__state = state
        self.variables = self.__get_variables(state, variable_names)

    def frame(self): return self.__state.screen_buffer

    def var(self, name): return self.variables[name]

    @staticmethod
    def __get_variables(state, variable_names):
        variables = {}
        for index in range(len(variable_names)):
            variables[variable_names[index]] = state.game_variables[index]
        return variables
