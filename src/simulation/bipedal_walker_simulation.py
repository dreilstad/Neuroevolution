import numpy as np
import gymnasium as gym
from simulation.simulator import Simulator


class BipedalWalkerSimulator(Simulator):

    def __init__(self, objectives):
        super().__init__(objectives)
        self.env = gym.make("BipedalWalker-v3")
        self.env.reset()
        self.domain = "bipedal"
        # 24 input nodes
        # 4 output nodes

    def simulate(self, neural_network):

        all_activations = None
        if self.CKA is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        task_performance = 0
        action = np.array([0.0, 0.0, 0.0, 0.0])

        terminated = False
        truncated = False

        while (not terminated) and (not truncated):

            # step
            state, reward, terminated, truncated, info = self.env.step(action)
            task_performance += reward

            # activate
            action, activations = neural_network.activate(state)

            # save sequence if using hamming distance
            if self.hamming is not None:
                sequence.extend([*state, *action])

            # append activations if using CKA
            if self.CKA is not None:
                all_activations.append(activations)

        novelty = self._get_novelty_characteristic(neural_network)

        # [performance, hamming, novelty, CKA, Q]
        return {"performance": task_performance,
                "hamming": self._binarize_sequence(sequence),
                "novelty": novelty,
                "CKA": all_activations}

    def _get_novelty_characteristic(self, neural_network):
        # TODO: change novelty to use output from fixed input
        return [*self.env.hull.position]