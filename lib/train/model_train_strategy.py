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
            gamma
    ):
        self.bach_size = batch_size
        self.model = model
        self.target_model = target_model
        self.state_transitions_memory = state_transitions_memory
        self.time_step_per_train = time_step_per_train
        self.input_shape = input_shape
        self.gamma = gamma

    def train(self):
        samples_count = self.exampels_count()
        state_transitions = self.state_transitions_memory.samples(samples_count)
        start_states, end_states = self.zeros_input(samples_count), self.zeros_input(samples_count)
        actions, rewards, final = [], [], []

        for index in range(samples_count):
            start_states[index, :, :, :] = state_transitions[index].start_state
            end_states[index, :, :, :] = state_transitions[index].end_state
            actions.append(state_transitions[index].action)
            rewards.append(state_transitions[index].rewards)
            final.append(state_transitions[index].final)

        actions_q_values = self.model.predict(start_states)
        next_actions_q_values = self.model.predict(end_states)
        target_next_actions_q_values = self.target_model.predict(end_states)

        for index in range(samples_count):
            actions_q_values[index][actions[index]] = self.td_target(
                final[index],
                next_actions_q_values[index],
                rewards[index],
                target_next_actions_q_values[index],
                self.gamma
            )

        history = self.model.fit(start_states, actions_q_values, batch_size=samples_count, nb_epochs=1)

        loss = history.history['loss']
        last_max_o_value = np.argmax(next_actions_q_values[-1])
        return last_max_o_value, loss

    @staticmethod
    def td_target(done, next_actions_q_values, rewards, target_next_actions_q_values, gamma):
        if done:
            return rewards
        high_q_value_action = np.argmax(next_actions_q_values)
        target_q_value = target_next_actions_q_values[high_q_value_action]
        return rewards + gamma * target_q_value

    def zeros_input(self, count): return np.zero((count,) + self.input_shape)

    def exampels_count(self): return min(self.bach_size * self.time_step_per_train, len(self.state_transitions_memory))


