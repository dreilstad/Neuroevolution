from simulation.simulator import Simulator
from simulation.environments.retina.retina_environment import RetinaEnvironment, HardRetinaEnvironment, Side, VisualObject


class RetinaSimulator(Simulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        self.env = RetinaEnvironment()
        self.use_input_nodes_in_mod_div = True

    def simulate(self, neural_network):

        all_activations = None
        if self.CKA is not None or self.cos_sim is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        # Evaluate the neural network against all combinations of the left and the right visual objects
        # at correct and incorrect sides of retina
        error_sum = 0.0
        for left in self.env.visual_objects:
            for right in self.env.visual_objects:
                error, net_input, net_output, activations = self._evaluate(neural_network, left, right)
                error_sum += error

                # save sequence if using hamming distance
                if self.hamming is not None:
                    sequence.extend([*net_input, *net_output])

                # append activations if using CKA
                if self.CKA is not None or self.cos_sim is not None:
                    all_activations.append(activations)

        # calculate the task performance score
        task_performance = 1.0 - float(error_sum/self.env.N)

        # store network output of test patterns as genome's novelty characteristics
        novelty = self.env.get_novelty_characteristic(neural_network) if self.novelty is not None else None

        # [performance, hamming, novelty, CKA, Q]
        #return [task_performance, self._binarize_sequence(sequence), novelty, all_activations]
        return {"performance": task_performance,
                "hamming": self._binarize_sequence(sequence),
                "novelty": novelty,
                "activations": all_activations}

    @staticmethod
    def _evaluate(neural_network, left, right):
        """
        The function to evaluate ANN against specific visual objects at lEFT and RIGHT side
        """

        # prepare input
        inputs = left.get_data() + right.get_data()
        #inputs.append(0.5) # the bias

        # activate
        network_output, activations = neural_network.activate(inputs)

        # get outputs
        network_output[0] = 1.0 if network_output[0] >= 0.5 else 0.0
        network_output[1] = 1.0 if network_output[1] >= 0.5 else 0.0

        # set ground truth
        left_target = 1.0 if left.side == Side.LEFT or left.side == Side.BOTH else 0.0
        right_target = 1.0 if right.side == Side.RIGHT or right.side == Side.BOTH else 0.0
        targets = [left_target, right_target]

        # find error as a distance between outputs and groud truth
        error = (network_output[0] - targets[0]) * (network_output[0] - targets[0]) + \
                (network_output[1] - targets[1]) * (network_output[1] - targets[1])

        return error, inputs, network_output, activations


class HardRetinaSimulator(RetinaSimulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        self.env = HardRetinaEnvironment()
