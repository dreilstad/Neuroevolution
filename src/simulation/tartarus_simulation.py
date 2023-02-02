import numpy as np
from simulation.simulator import Simulator
from simulation.environments.tartarus.tartarus_environment import TartarusEnvironment, DeceptiveTartarusEnvironment
from simulation.environments.tartarus.tartarus_util import generate_configurations


class TartarusSimulator(Simulator):

    def __init__(self, objectives, domain, N=6, K=6):
        super().__init__(objectives, domain)
        self.configurations, self.test_config, self.agent_pos, self.agent_dir = generate_configurations(N, K,
                                                                                                        sample=100)
        self.environment_cls = TartarusEnvironment
        self.use_input_nodes_in_mod_div = True
        # 8 input nodes (NW, N, NE, W, E, SW, S, SE) or
        # 11 input nodes (NW, N, NE, W, E, SW, S, SE, left, right, forward)
        # 3 output nodes (left, right, forward)

    def simulate(self, neural_network):

        all_activations = None
        if self.CKA is not None or self.cos_sim is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        print("Starting Tartarus simulation")
        task_performance = 0.0
        for i, configuration in enumerate(self.configurations):
            print(f"config {i}")
            env = self.environment_cls(configuration)
            state, _ = env.reset()
            state.extend([0, 0, 0])

            terminated = False
            truncated = False

            while (not terminated) and (not truncated):

                # activate network and get next action
                output, activations = neural_network.activate(state)
                action = np.argmax(output)

                # save sequence if using hamming distance
                if self.hamming is not None:
                    sequence.extend([*state, *output])

                # append activations if using CKA
                if self.CKA is not None or self.cos_sim is not None:
                    all_activations.append(activations)

                # step
                state, _, terminated, truncated, _ = env.step(action)

                # one-hot encode network output to use as input next iteration
                output_one_hot = [0, 0, 0]
                output_one_hot[action] = 1
                state.extend([output_one_hot])

            task_performance += env.state_evaluation()

        novelty = self._get_novelty_characteristic(neural_network)

        task_performance = task_performance / len(self.configurations)

        # [performance, hamming, novelty, CKA, Q]
        return {"performance": task_performance,
                "hamming": self._binarize_sequence(sequence),
                "novelty": novelty,
                "activations": all_activations}

    def _get_novelty_characteristic(self, neural_network):

        env = self.environment_cls(self.test_config,
                                   fixed_agent_pos=self.agent_pos,
                                   fixed_agent_dir=self.agent_dir)
        state, _ = env.reset()
        state.extend([0, 0, 0])

        terminated = False
        truncated = False
        while (not terminated) and (not truncated):

            # activate network and get next action
            output, activations = neural_network.activate(state)
            action = np.argmax(output)

            # step
            state, _, terminated, truncated, _ = env.step(action)

            # one-hot encode network output to use as input next iteration
            output_one_hot = [0, 0, 0]
            output_one_hot[action] = 1
            state.extend([output_one_hot])

        tartarus_state = env.encode_tartarus_state()

        return list(tartarus_state.ravel())


class DeceptiveTartarusSimulator(TartarusSimulator):

    def __init__(self, objectives, N=6, K=6):
        super().__init__(
            objectives=objectives,
            N=N,
            K=K
        )

        self.environment_cls = DeceptiveTartarusEnvironment

