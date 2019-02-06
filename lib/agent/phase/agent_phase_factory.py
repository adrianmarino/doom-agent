from lib.agent.phase.impl.agent_explore_phase import AgentExplorePhase
from lib.agent.phase.impl.agent_final_phase import AgentFinalPhase
from lib.agent.phase.impl.agent_observe_phase import AgentObservePhase


class AgentPhaseFactory:
    @staticmethod
    def create(ctx):
        if ctx.time <= ctx.observe_times:
            return AgentObservePhase()
        elif ctx.observe_times < ctx.time <= (ctx.observe_times + ctx.explore_times):
            return AgentExplorePhase()
        else:
            return AgentFinalPhase()
