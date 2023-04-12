import numpy as np
from simulation.simulator import Simulator
from simulation.environments.tartarus.tartarus_environment import TartarusEnvironment, DeceptiveTartarusEnvironment
from simulation.environments.tartarus.tartarus_util import generate_configurations


class TartarusSimulator(Simulator):

    def __init__(self, objectives, domain, N=6, K=6):
        super().__init__(objectives, domain)

        configs, agent_states = generate_configurations(N, K, sample=20)
        self.configurations = configs
        self.initial_agent_states = agent_states

        self.environment_cls = TartarusEnvironment
        self.use_input_nodes_in_mod_div = True
        # 8 input nodes (NW, N, NE, W, E, SW, S, SE) or
        # 11 input nodes (NW, N, NE, W, E, SW, S, SE, left, right, forward)
        # 3 output nodes (left, right, forward)

        low = np.array([-1.0] * 8 + [0.0] * 3).astype(np.float32)
        high = np.array([1.1] * 8 + [1.0] * 3).astype(np.float32)
        self.input_value_range = list(zip(low, high))

    def simulate(self, neural_network):

        all_activations = None
        if self.CKA is not None or self.CCA is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        behavior = None
        if self.novelty is not None:
            behavior = []

        task_performance = 0.0
        for configuration, (agent_pos, agent_dir) in zip(self.configurations, self.initial_agent_states):
            env = self.environment_cls(configuration,
                                       fixed_agent_pos=agent_pos,
                                       fixed_agent_dir=agent_dir)
            state, _ = env.reset()
            state = np.concatenate((state, [0, 0, 0]))

            terminated = False
            truncated = False

            while (not terminated) and (not truncated):

                # activate network and get next action
                output, activations = neural_network.activate(state)
                action = np.argmax(output)

                # save sequence if using hamming distance
                if self.hamming is not None:
                    # normalize input values to the range [0, 1]
                    norm_state = self._normalize_sequence(state, self.input_value_range)
                    sequence.extend([*norm_state, *output])

                # append activations if using CKA or CCA
                if self.CKA is not None or self.CCA is not None:
                    all_activations.append(activations)

                # step
                state, _, terminated, truncated, _ = env.step(action)

                # one-hot encode network output to use as input next iteration
                output_one_hot = [0, 0, 0]
                output_one_hot[action] = 1
                state = np.concatenate((state, output_one_hot))

            if self.novelty is not None:
                behavior.append(env.get_final_block_positions_vector())

            task_performance += env.state_evaluation()

        task_performance = task_performance / len(self.configurations)

        novelty = behavior if self.novelty is not None else None

        # store Q-score if using modularity objective
        q_score = self.mod(neural_network.all_nodes, neural_network.all_connections) if self.mod is not None else 0.0

        # [performance, hamming, novelty, CKA, Q]
        return {"performance": task_performance,
                "hamming": self._binarize_sequence(sequence),
                "novelty": novelty,
                "activations": all_activations,
                "Q": q_score}

    def _get_novelty_characteristic(self, neural_network):
        pass


class DeceptiveTartarusSimulator(TartarusSimulator):

    def __init__(self, objectives, domain, N=6, K=6):
        super().__init__(
            objectives=objectives,
            domain=domain,
            N=N,
            K=K
        )

        self.environment_cls = DeceptiveTartarusEnvironment

