from lib.agent.phase.agent_phase import AgentPhase


@AgentPhase.register
class AgentFinalPhase:
    def on_each_time(self, ctx):
        ctx.log('Final', 'End training')

    def on_episode_finish(self, ctx):
        pass
