import numpy as np
from simulation.simulator import Simulator
from simulation.environments.retina.retina_environment import RetinaEnvironment, HardRetinaEnvironment, \
                                                              HardRetinaEnvironmentExtended, Side, VisualObject


class RetinaSimulator(Simulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        self.env = RetinaEnvironment()
        self.use_input_nodes_in_mod_div = True

        self.output_value_range = [(-1.0, 1.0)]

    def simulate(self, neural_network):

        all_activations = None
        if self.CKA is not None or self.CCA is not None:
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
                    # normalize output values to the range [0, 1]
                    norm_net_output = self._normalize_sequence(net_output, self.output_value_range)
                    sequence.extend([*net_input, *norm_net_output])

                # append activations if using CKA or CCA
                if self.CKA is not None or self.CCA is not None:
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

        # activate
        network_output, activations = neural_network.activate(inputs)
        """
        # get outputs
        left_output = 1 if network_output[0] >= 0.5 else 0
        right_output = 1 if network_output[1] >= 0.5 else 0

        # set ground truth
        left_target = 1 if left.side == Side.LEFT or left.side == Side.BOTH else 0
        right_target = 1 if right.side == Side.RIGHT or right.side == Side.BOTH else 0

        # left AND right operation, error is 0 if true, error is 1 if false
        error = float(not(left_output == left_target and right_output == right_target))
        """

        # output classification
        output = 1 if network_output[0] > 0.0 else 0

        # get target classification
        left_target = 1 if left.side == Side.LEFT or left.side == Side.BOTH else 0
        right_target = 1 if right.side == Side.RIGHT or right.side == Side.BOTH else 0
        target = int(left_target and right_target)

        # get error of classification
        error = float(not(output == target))

        return error, inputs, network_output, activations


class HardRetinaSimulator(RetinaSimulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        self.env = HardRetinaEnvironment()


class HardRetinaExtendedSimulator(RetinaSimulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        self.env = HardRetinaEnvironmentExtended()

    def simulate(self, neural_network):

        all_activations = None
        if self.CKA is not None or self.CCA is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        # Evaluate the neural network against all combinations of the left and the right visual objects
        # at correct and incorrect sides of retina
        error_sum = 0.0
        for left in self.env.visual_objects:
            for middle in self.env.visual_objects:
                for right in self.env.visual_objects:
                    error, net_input, net_output, activations = self._evaluate(neural_network, left, middle, right)
                    error_sum += error

                    # save sequence if using hamming distance
                    if self.hamming is not None:
                        sequence.extend([*net_input, *net_output])

                    # append activations if using CKA or CCA
                    if self.CKA is not None or self.CCA is not None:
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
    def _evaluate(neural_network, left, middle, right):
        """
        The function to evaluate ANN against specific visual objects at lEFT and RIGHT side
        """

        # prepare input
        inputs = left.get_data() + middle.get_data() + right.get_data()

        # activate
        network_output, activations = neural_network.activate(inputs)

        # get outputs
        left_output = 1 if network_output[0] >= 0.5 else 0
        middle_output = 1 if network_output[1] >= 0.5 else 0
        right_output = 1 if network_output[2] >= 0.5 else 0

        # set ground truth
        left_target = 1 if left.side == Side.LEFT else 0
        middle_target = 1 if middle.side == Side.MIDDLE else 0
        right_target = 1 if right.side == Side.RIGHT else 0

        # left AND right operation, error is 0 if true, error is 1 if false
        error = float(not(left_output == left_target and middle_output == middle_target and right_output == right_target))

        return error, inputs, network_output, activations
