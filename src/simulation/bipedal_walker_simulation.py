import math
import numpy as np
import gymnasium as gym
from simulation.simulator import Simulator


class BipedalWalkerSimulator(Simulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        self.env = gym.make("BipedalWalker-v3")
        self.use_input_nodes_in_mod_div = True
        # 24 input nodes
        # 4 output nodes

        low = np.array([-math.pi, -5.0, -5.0, -5.0, -math.pi, -5.0, -math.pi,
                        -5.0, -0.0, -math.pi, -5.0, -math.pi, -5.0, -0.0] + [-1.0] * 10).astype(np.float32)
        high = np.array([math.pi, 5.0, 5.0, 5.0, math.pi, 5.0, math.pi,
                         5.0, 5.0, math.pi, 5.0, math.pi, 5.0, 5.0] + [1.0] * 10).astype(np.float32)
        self.input_value_range = list(zip(low, high))

        low = np.array([-1.0] * 4).astype(np.float32)
        high = np.array([1.0] * 4).astype(np.float32)
        self.output_value_range = list(zip(low, high))

    def simulate(self, neural_network):

        self.env.reset()

        all_activations = None
        if self.CKA is not None or self.CCA is not None:
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
                # normalize input and output values to the range [0, 1]
                norm_state = self._normalize_sequence(state, self.input_value_range)
                norm_action = self._normalize_sequence(action, self.output_value_range)
                sequence.extend([*norm_state, *norm_action])

            # append activations if using CKA or CCA
            if self.CKA is not None or self.CCA is not None:
                all_activations.append(activations)

        novelty = self._get_novelty_characteristic(neural_network) if self.novelty is not None else None

        # [performance, hamming, novelty, CKA, Q]
        return {"performance": task_performance,
                "hamming": self._binarize_sequence(sequence),
                "novelty": novelty,
                "activations": all_activations}

    def _get_novelty_characteristic(self, neural_network):

        # behavior vector
        behavior = []

        # for each input node, we set input to 1 and 0 for the rest and record the network output
        state = [0.0] * len(neural_network.input_nodes)
        for i in range(len(state)):
            state[i] = 1.0
            network_output, _ = neural_network.activate(state)
            behavior.extend(network_output)
            state[i] = 0.0

        return behavior
