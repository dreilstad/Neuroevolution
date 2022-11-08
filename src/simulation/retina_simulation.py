from util import binarize_sequence
from simulation.simulator import Simulator
from simulation.environments.retina.retina_environment import RetinaEnvironment, Side
from objectives.novelty import NoveltyItem


class RetinaSimulator(Simulator):

    def __init__(self, objectives):
        self.metric = self.novelty_metric
        super().__init__(objectives)
        self.env = RetinaEnvironment()

    def simulate(self, genome_id, genome, neural_network, generation):
        error_sum = 0.0

        sequence = None
        if self.hamming is not None:
            sequence = []

        nov_item = None
        if self.novelty is not None:
            nov_item = NoveltyItem(generation=generation, genome_id=genome_id)
            self.novelty_items[genome_id] = nov_item

        # Evaluate the detector ANN against 256 combintaions of the left and the right visual objects
        # at correct and incorrect sides of retina
        for left in self.env.visual_objects:
            for right in self.env.visual_objects:
                error, network_input, network_output = RetinaSimulator._evaluate(neural_network, left, right)
                error_sum += error

                # save sequence if using hamming distance
                if self.hamming is not None:
                    bin_sequence = binarize_sequence([*network_input, *network_output])
                    sequence.extend(bin_sequence)

                # add to novelty item if using behavioural diversity
                if self.novelty is not None:
                    nov_item.data.extend(network_output)
        
        # calculate the task performance score
        task_performance = 1.0 - float(error_sum/256.0)

        return [task_performance, sequence]

    @staticmethod
    def _evaluate(neural_network, left, right):
        """
        The function to evaluate ANN against specific visual objects at lEFT and RIGHT side
        """

        # prepare input
        inputs = left.get_data() + right.get_data()
        inputs.append(0.5) # the bias

        # activate
        # TODO: "activations" output is currently None, change nn to also return all activations
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

        return error, inputs, network_output

    @staticmethod
    def novelty_metric(first_item, second_item):
        if not (hasattr(first_item, "data") or hasattr(second_item, "data")):
            return NotImplemented

        if len(first_item.data) != len(second_item.data):
            # can not be compared
            return 0.0

        diff_accum = 0.0
        size = len(first_item.data)
        for i in range(size):
            diff = abs(first_item.data[i] - second_item.data[i])
            diff_accum += diff

        return diff_accum / float(size)

if __name__=="__main__":
    r = RetinaSimulator("fitness")
