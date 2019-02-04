import numpy as np


class ModelTrainStrategy:
    def __init__(
            self,
            model,
            target_model,
            state_transitions_memory,
            batch_size,
            time_step_per_train,
            input_shape,
            gamma,
            model_input_converter,
            callbacks
    ):
        self.__bach_size = batch_size
        self.__model = model
        self.__target_model = target_model
        self.__state_transitions_memory = state_transitions_memory
        self.__time_step_per_train = time_step_per_train
        self.__input_shape = input_shape
        self.__gamma = gamma
        self.__model_input_converter = model_input_converter
        self.__callbacks = callbacks

    def train(self):
        samples_count = self.examples_count()
        state_transitions = self.__state_transitions_memory.samples(samples_count)
        start_states, end_states = self.zeros_input(samples_count), self.zeros_input(samples_count)
        actions, rewards, final = [], [], []

        for index in range(samples_count):
            start_states[index, :, :, :] = self.__to_input(state_transitions[index].start_state)
            end_states[index, :, :, :] = self.__to_input(state_transitions[index].end_state)
            actions.append(state_transitions[index].action)
            rewards.append(state_transitions[index].rewards)
            final.append(state_transitions[index].final)

        actions_q_values = self.__model.predict(start_states)
        next_actions_q_values = self.__model.predict(end_states)
        target_next_actions_q_values = self.__target_model.predict(end_states)

        for index in range(samples_count):
            actions_q_values[index][actions[index]] = self.td_target(
                final[index],
                next_actions_q_values[index],
                rewards[index],
                target_next_actions_q_values[index],
                self.__gamma
            )

        history = self.__model.fit(
            start_states,
            actions_q_values,
            batch_size=samples_count,
            callbacks=self.__callbacks
        )

        return history.history['loss'][0]

    def __to_input(self, frames):
        return self.__model_input_converter.convert(frames)

    @staticmethod
    def td_target(done, next_actions_q_values, rewards, target_next_actions_q_values, gamma):
        if done:
            return rewards
        high_q_value_action = np.argmax(next_actions_q_values)
        target_q_value = target_next_actions_q_values[high_q_value_action]
        return rewards + gamma * target_q_value

    def zeros_input(self, count):
        return np.zeros((count,) + self.__input_shape.as_tuple())

    def examples_count(self):
        return min(self.__bach_size * self.__time_step_per_train, len(self.__state_transitions_memory))
