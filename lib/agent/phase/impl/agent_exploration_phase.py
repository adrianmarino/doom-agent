from lib.agent.phase.agent_phase import AgentPhase


@AgentPhase.register
class AgentExplorationPhase:
    def __init__(self, agent, train_freq, update_target_model_freq):
        self.__agent = agent
        self.__train_freq = train_freq
        self.__update_target_model_freq = update_target_model_freq

    def perform(self, time, episode):
        self.__agent.epsilon.decrement(time)

        if time % self.__train_freq == 0:
            loss = self.__agent.model_train_strategy.train()
            self.__agent.logger.info(f'Time:{time} - Episode:{episode} - Phase:Explore - Epsilon:{self.__agent.epsilon.value()}')

        if time % self.__update_target_model_freq == 0:
            self.__agent.model.copy_weights_to(self.__agent.target_model)
            self.__agent.logger.info(f'Time:{time} - Episode:{episode} - Phase:Explore - Update TD Target Model (Each {self.__update_target_model_freq} times)')

        self.__agent.exec_callbacks(time, episode)
