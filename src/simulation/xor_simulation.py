from .simulator import Simulator
from util import binarize_sequence


class XORSimulator(Simulator):

    def __init__(self, objectives):
        super().__init__(objectives)
        self.xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        self.xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
            
    def simulate(self, genome_id, genome, neural_network, generation):

        sequence = None
        if self.hamming is not None:
            sequence = []
            
        task_performance = 4.0

        for xor_input, xor_output in zip(self.xor_inputs, self.xor_outputs):
            # TODO: "activations" output is currently None, change nn to also return all activations
            network_output, activations = neural_network.activate(xor_input)
            task_performance -= (network_output[0] - xor_output[0]) ** 2

            # save sequence if using hamming distance
            if self.hamming is not None:
                bin_sequence = binarize_sequence([*xor_input, *network_output])
                sequence.extend(bin_sequence)

        return [task_performance, sequence]
