import copy
import numpy as np
import gym
from simulation.simulator import Simulator


class BipedalWalkerSimulator(Simulator):

    def __init__(self, objectives):
        super().__init__(objectives)
        self.env = gym.make("BipedalWalker-v3")
        # 24 input nodes
        # 4 output nodes

    def simulate(self, genome_id, genome, neural_network, generation):
        env = copy.deepcopy(self.env)
        env.reset()

        all_activations = None
        if self.CKA is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        steps = 0
        task_performance = 0
        action = np.array([0.0, 0.0, 0.0, 0.0])

        terminated = False
        truncated = False

        while (not terminated) and (not truncated):
            state, reward, terminated, truncated, info = env.step(action)
            task_performance += reward
            steps += 1

            """
            print(f"action: {action}")
            print(f"state: {state}\n reward: {reward}\n terminated: {terminated}\n truncated: {truncated}\n info: {info}")
            print(f"task_performance: {task_performance}\n")
            """

            action, activations = neural_network.activate(state)

            # save sequence if using hamming distance
            if self.hamming is not None:
                sequence.extend([*state, *action])

            # append activations if using CKA
            if self.CKA is not None:
                all_activations.append(activations)

        novelty = [*env.hull.position]

        # [performance, hamming, novelty, CKA, Q]
        return [task_performance, self._binarize_sequence(sequence), novelty, all_activations]
