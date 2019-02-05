from lib.agent.phase.agent_phase import AgentPhase


@AgentPhase.register
class AgentObservationPhase:
    def __init__(self, agent):
        self.__agent = agent

    def on_episode_finish(self, time, episode):
        self.__agent.logger.info(
            f'Time:{time} - Episode:{episode} - OBSERVE - Env:{self.__agent.env.previous_state().variables}'
        )

    def each_time(self, time, episode):
        pass