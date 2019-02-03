class DoomRewardsComputationStrategy:
    def calculate(self, env, current_state, next_state):
        rewards = env.accumulated_rewards()
        if env.is_episode_finished():
            return rewards

        if current_state.var('kills') > next_state.var('kills'):
            rewards += 1

        if current_state.var('ammo') < next_state.var('ammo'):
            rewards -= 1

        if current_state.var('health') < next_state.var('health'):
            rewards -= 1

        return rewards
