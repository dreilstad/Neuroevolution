import numpy as np
from simulation.simulator import Simulator
from simulation.environments.tartarus.tartarus_environment import TartarusEnvironment
from simulation.environments.tartarus.tartarus_util import generate_configurations


class TartarusSimulator(Simulator):

    def __init__(self, objectives, N=6, K=6):
        super().__init__(objectives)
        self.configurations, self.test_config = generate_configurations(N, K, sample_size=1000)
        # 8 input nodes (NW, N, NE, W, E, SW, S, SE) or
        # 11 input nodes (NW, N, NE, W, E, SW, S, SE, left, right, forward)
        # 3 output nodes (left, right, forward)

    def simulate(self, neural_network):

        all_activations = None
        if self.CKA is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        print("Starting Tartarus simulation")
        task_performance = 0.0
        for i, configuration in enumerate(self.configurations):
            print(f"config {i}")
            env = TartarusEnvironment(configuration)
            state, _ = env.reset()

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
                if self.CKA is not None:
                    all_activations.append(activations)

                # step
                state, reward, terminated, truncated, info = env.step(action)

            task_performance += env.state_evaluation()

        novelty = self._get_novelty_characteristic(neural_network)

        task_performance = task_performance / len(self.configurations)

        # [performance, hamming, novelty, CKA, Q]
        return [task_performance, self._binarize_sequence(sequence), novelty, all_activations]

    def _get_novelty_characteristic(self, neural_network):

        env = TartarusEnvironment(self.test_config)
        state, _ = env.reset()

        terminated = False
        truncated = False
        while (not terminated) and (not truncated):

            # activate network and get next action
            output, activations = neural_network.activate(state)
            action = np.argmax(output)

            # step
            state, reward, terminated, truncated, info = env.step(action)

        tartarus_state = env.encode_tartarus_state()

        return tartarus_state.ravel()