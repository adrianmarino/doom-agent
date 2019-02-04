from lib.agent.phase.impl.agent_exploration_phase import AgentExplorationPhase
from lib.agent.phase.impl.agent_final_phase import AgentFinalPhase
from lib.agent.phase.impl.agent_observation_phase import AgentObservationPhase


class AgentPhaseFactory:
    def __init__(self, agent, observe_times, explore_times, train_freq, update_target_model_freq):
        self.__observe_times = observe_times
        self.__explore_times = explore_times

        self.observe_phase = AgentObservationPhase(agent)
        self.explore_phase = AgentExplorationPhase(agent, train_freq, update_target_model_freq)
        self.final_phase = AgentFinalPhase(agent)

    def create(self, agent, time):
        if time <= self.__observe_times:
            return self.observe_phase
        elif self.__observe_times < time <= (self.__observe_times + self.__explore_times):
            return self.explore_phase
        else:
            return self.final_phase
