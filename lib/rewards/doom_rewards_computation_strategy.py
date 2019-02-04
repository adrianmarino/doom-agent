class DoomRewardsComputationStrategy:
    def calculate(self, env, initial_state, final_state):
        rewards = env.accumulated_rewards()

        if env.is_episode_finished():
            return rewards

        if initial_state.var('kills') > final_state.var('kills'):
            rewards += 1

        if initial_state.var('ammo') < final_state.var('ammo'):
            rewards -= 1

        if initial_state.var('health') < final_state.var('health'):
            rewards -= 1

        return rewards
