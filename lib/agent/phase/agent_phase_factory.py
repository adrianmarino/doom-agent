from lib.agent.phase.impl.agent_explore_phase import AgentExplorePhase
from lib.agent.phase.impl.agent_final_phase import AgentFinalPhase
from lib.agent.phase.impl.agent_observe_phase import AgentObservePhase
from lib.agent.phase.impl.agent_play_phase import AgentPlayPhase


class AgentPhaseFactory:
    def __init__(self):
        self.__observe_phase = AgentObservePhase()
        self.__explore_phase = AgentExplorePhase()
        self.__final_phase = AgentFinalPhase()
        self.__play_phase = AgentPlayPhase()

    def create(self, ctx):
        if ctx.time <= ctx.observe_times:
            return self.__observe_phase
        elif ctx.observe_times < ctx.time <= (ctx.observe_times + ctx.explore_times):
            return self.__explore_phase
        elif ctx.time > (ctx.observe_times + ctx.explore_times):
            return self.__final_phase
        else:
            return self.__play_phase
