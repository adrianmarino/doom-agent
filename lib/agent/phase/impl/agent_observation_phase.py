from lib.agent.phase.agent_phase import AgentPhase


@AgentPhase.register
class AgentObservationPhase:
    def __init__(self, agent):
        self.__agent = agent

    def perform(self, time, episode):
        self.__agent.logger.info(
            f'Time:{time} - Episode:{episode} - Observe - Env:{self.__agent.env.current_state().variables}'
        )
