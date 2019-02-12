from lib.train.algorithm.ddqn.phase.impl.algorithm_explore_phase import AlgorithmExplorePhase
from lib.train.algorithm.ddqn.phase.impl.algorithm_final_phase import AlgorithmFinalPhase
from lib.train.algorithm.ddqn.phase.impl.algorithm_observe_phase import AlgorithmObservePhase


class AlgorithmPhaseFactory:
    def __init__(self):
        self.__observe_phase = AlgorithmObservePhase()
        self.__explore_phase = AlgorithmExplorePhase()
        self.__final_phase = AlgorithmFinalPhase()

    def create(self, ctx):
        if ctx.time <= ctx.observe_times:
            return self.__observe_phase
        elif ctx.observe_times < ctx.time <= (ctx.observe_times + ctx.explore_times + ctx.train_times):
            return self.__explore_phase
        elif ctx.time > (ctx.observe_times + ctx.explore_times + ctx.train_times):
            return self.__final_phase
