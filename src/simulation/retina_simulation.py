from simulation.simulator import Simulator
from simulation.environments.retina.retina_environment import RetinaEnvironment, Side, VisualObject


class RetinaSimulator(Simulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        self.env = RetinaEnvironment()
        self.use_input_nodes_in_mod_div = True

    def simulate(self, neural_network):

        all_activations = None
        if self.CKA is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        # Evaluate the detector ANN against 256 combintaions of the left and the right visual objects
        # at correct and incorrect sides of retina
        error_sum = 0.0
        for left in self.env.visual_objects:
            for right in self.env.visual_objects:
                error, net_input, net_output, activations = RetinaSimulator._evaluate(neural_network, left, right)
                error_sum += error

                # save sequence if using hamming distance
                if self.hamming is not None:
                    sequence.extend([*net_input, *net_output])

                # append activations if using CKA
                if self.CKA is not None:
                    all_activations.append(activations)

        # calculate the task performance score
        task_performance = 1.0 - float(error_sum/256.0)

        # store network output of test patterns as genome's novelty characteristics
        novelty = self._get_novelty_characteristic(neural_network)

        # [performance, hamming, novelty, CKA, Q]
        #return [task_performance, self._binarize_sequence(sequence), novelty, all_activations]
        return {"performance": task_performance,
                "hamming": self._binarize_sequence(sequence),
                "novelty": novelty,
                "CKA": all_activations}

    @staticmethod
    def _evaluate(neural_network, left, right):
        """
        The function to evaluate ANN against specific visual objects at lEFT and RIGHT side
        """

        # prepare input
        inputs = left.get_data() + right.get_data()
        inputs.append(0.5) # the bias

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

    def _get_novelty_characteristic(self, neural_network):

        # behavior vector
        behavior = []

        # test patterns to feed through the network to get behavior
        test_patterns = [(VisualObject("o .\n. o", side=Side.LEFT), VisualObject(". o\no .", side=Side.RIGHT)),
                         (VisualObject(". o\no .", side=Side.LEFT), VisualObject("o .\n. o", side=Side.RIGHT)),
                         (VisualObject("o o\no o", side=Side.LEFT), VisualObject(". .\n. .", side=Side.RIGHT)),
                         (VisualObject(". .\n. .", side=Side.LEFT), VisualObject("o o\no o", side=Side.RIGHT))]

        # iterate test patterns
        for test_pattern_left, test_pattern_right in test_patterns:

            test_input = test_pattern_left.get_data() + test_pattern_right.get_data()
            test_input.append(0.5)

            network_output, _ = neural_network.activate(test_input)
            behavior.extend(network_output)

        return behavior
