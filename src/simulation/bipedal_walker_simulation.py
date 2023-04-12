import numpy as np
import gymnasium as gym
from simulation.simulator import Simulator


class BipedalWalkerSimulator(Simulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        self.env = gym.make("BipedalWalker-v3")
        self.env.reset()
        self.initial_hull_position = tuple(self.env.hull.position)
        self.behavioral_sample_rate = 16

        self.use_input_nodes_in_mod_div = True
        # 24 input nodes
        # 4 output nodes

        low = np.array([-np.pi, -5.0, -5.0, -5.0, -np.pi, -5.0, -np.pi,
                        -5.0, 0.0, -np.pi, -5.0, -np.pi, -5.0, 0.0] + [-1.0] * 10).astype(np.float32)
        high = np.array([np.pi, 5.0, 5.0, 5.0, np.pi, 5.0, np.pi,
                         5.0, 1.0, np.pi, 5.0, np.pi, 5.0, 1.0] + [1.0] * 10).astype(np.float32)
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

        behavior = None
        if self.novelty is not None:
            behavior = []

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

            if self.novelty is not None:
                if self.env._elapsed_steps % self.behavioral_sample_rate == 0:
                    behavior.append(tuple(self.env.hull.position))

        # get novelty characteristic
        novelty = self._get_novelty_characteristic(behavior) if self.novelty is not None else None

        # store Q-score if using modularity objective
        q_score = self.mod(neural_network.all_nodes, neural_network.all_connections) if self.mod is not None else 0.0

        # [performance, hamming, novelty, CKA, Q]
        return {"performance": task_performance,
                "hamming": self._binarize_sequence(sequence),
                "novelty": novelty,
                "activations": all_activations,
                "Q": q_score}

    def _get_novelty_characteristic(self, behavior):
        """ Behavior is defined as the sampled offset of
        """
        behavior_vector = []
        for pos in behavior:
            x_offset = np.sign(pos[0] - self.initial_hull_position[0]) * (pos[0] - self.initial_hull_position[0])**2
            y_offset = np.sign(pos[1] - self.initial_hull_position[1]) * (pos[1] - self.initial_hull_position[1])**2
            behavior_vector.append(x_offset)
            behavior_vector.append(y_offset)

        # if an individual falls, the behavior vector is extended with last behavior to max size
        # max behavior vector size is the max steps divided by a set sampling rate
        max_behavior_vector_size = 2 * (self.env._max_episode_steps // self.behavioral_sample_rate)
        if len(behavior_vector) < max_behavior_vector_size:
            last_behavior = [behavior_vector[-2], behavior_vector[-1]]
            extend_with_size = (max_behavior_vector_size - len(behavior_vector)) // 2
            behavior_vector.extend(last_behavior * extend_with_size)

        return behavior_vector
