from lib.agent.phase.agent_phase import AgentPhase


@AgentPhase.register
class AgentExplorePhase:
    def on_each_time(self, ctx):
        ctx.update_epsilon()

        if ctx.is_train_time():
            loss = ctx.train_model()

        if ctx.is_td_target_update_time():
            ctx.update_td_target_model()
            ctx.log('Explore', f'Update TD Target Model (Each {ctx.update_target_model_freq} times)')

    def on_episode_finish(self, ctx):
        ctx.log('Explore', f'Env:{ctx.previous_state_variables()} - Epsilon:{ctx.epsilon_value()}')
