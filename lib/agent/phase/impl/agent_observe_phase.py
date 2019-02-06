from lib.agent.phase.agent_phase import AgentPhase


@AgentPhase.register
class AgentObservePhase:
    def on_each_time(self, ctx):
        pass

    def on_episode_finish(self, ctx):
        ctx.log('Observe', f'Env:{ctx.previous_state_variables()}')
