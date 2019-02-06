class DoomRewardsComputationStrategy:
    def __init__(self, kills_rewards, ammo_rewards, health_rewards):
        self.__kills_rewards = kills_rewards
        self.__ammo_rewards = ammo_rewards
        self.__health_rewards = health_rewards

    def calculate(self, env, initial_state, final_state):
        rewards = env.accumulated_rewards()

        if env.is_episode_finished():
            return rewards

        if initial_state.var('kills') > final_state.var('kills'):
            rewards += self.__kills_rewards

        if initial_state.var('ammo') < final_state.var('ammo'):
            rewards += self.__ammo_rewards

        if initial_state.var('health') < final_state.var('health'):
            rewards += self.__health_rewards

        return rewards
