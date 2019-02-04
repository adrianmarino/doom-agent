from lib.agent.phase.agent_phase import AgentPhase


@AgentPhase.register
class AgentFinalPhase:
    def __init__(self, agent):
        self.__agent = agent

    def perform(self, time, episode):
        self.__agent.logger.info(f'Time:{time} - Episode:{episode} - Phase:Final - Finish training')
