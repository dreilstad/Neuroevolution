#
# The script to maintain the modular retina test environment.
# small modification to this text
# https://github.com/qendro/CH8/blob/f40a8b4fc066e74c903a42612b2862067e069b93/retina_experiment.py

from enum import Enum
import numpy as np

class Side(Enum):
    LEFT = 1
    RIGHT = 2
    BOTH = 3

class VisualObject:
    """
    The class to encode the visual object representation
    """
    def __init__(self, configuration, side, size=2):
        """
        Creates new instance with provided configuration and object size
        Arguments:
            configuration:  The configuration of the visual object in form of the text:
                            o o
                            o .
            side:           The side of the retina this object must occupy
            size:           The size of the visual object
        """
        self.size = size
        self.side = side
        self.configuration = configuration
        self.data = np.zeros((size, size))
        
        # Parse configuration
        lines = self.configuration.splitlines()
        for r, line in enumerate(lines):
            chars = line.split(" ")
            for c, ch in enumerate(chars):
                if ch == 'o':
                    # pixel is ON
                    self.data[r, c] = 1.0
                else:
                    # pixel is OFF
                    self.data[r, c] = 0.0

    def get_data(self):
        return self.data.flatten().tolist()
    
    def __str__(self):
        """
        Returns the nicely formatted string representation of this object.
        """
        return "%s\n%s" % (self.side.name, self.configuration)

class RetinaEnvironment:
    """
    Represents the modular retina environment holding test data set and providing
    methods to evaluate detector ANN against it.
    """
    def __init__(self):
        self.visual_objects = []

        # populate data set
        self.create_data_set()
        self.N = float(len(self.visual_objects) * len(self.visual_objects))

    def create_data_set(self):
        # set left side objects
        self.visual_objects.append(VisualObject(". .\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject(". .\n. o", side=Side.BOTH))
        self.visual_objects.append(VisualObject(". o\n. o", side=Side.LEFT))
        self.visual_objects.append(VisualObject(". o\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject(". o\no o", side=Side.LEFT))
        self.visual_objects.append(VisualObject(". .\no .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o o\n. o", side=Side.LEFT))
        self.visual_objects.append(VisualObject("o .\n. .", side=Side.BOTH))

        # set right side objects
        self.visual_objects.append(VisualObject(". .\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o .\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o .\no .", side=Side.RIGHT))
        self.visual_objects.append(VisualObject(". .\no .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o o\no .", side=Side.RIGHT))
        self.visual_objects.append(VisualObject(". o\n. .", side=Side.BOTH))
        self.visual_objects.append(VisualObject("o .\no o", side=Side.RIGHT))
        self.visual_objects.append(VisualObject(". .\n. o", side=Side.BOTH))

    def get_novelty_characteristic(self, neural_network):

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

            network_output, _ = neural_network.activate(test_input)
            behavior.extend(network_output)

        return behavior

    def __str__(self):
        """
        Returns the nicely formatted string representation of this environment.
        """
        str = "Retina Environment"
        for obj in self.visual_objects:
            str += "\n%s" % obj

        return str


class HardRetinaEnvironment(RetinaEnvironment):

    def __init__(self):
        super().__init__()

    def create_data_set(self):
        # set left side objects
        self.visual_objects.append(VisualObject(". . .\n. . .\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject("o . .\n. . .\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . o\n. . .\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\no . .\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\n. . o\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\n. . .\no . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\n. . .\n. . o", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\n. o o\n. o o", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject(". o o\n. o o\n. . .", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject(". o o\n. o o\n. o o", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject(". . o\n. . o\no o o", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject("o o o\n. . o\n. . o", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject(". . o\n. o o\no o o", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject("o o o\n. o o\n. . o", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject("o o .\no . .\n. . .", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject(". . .\no . .\no o .", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject(". o o\no o o\no o o", side=Side.LEFT, size=3))
        self.visual_objects.append(VisualObject("o o o\no o o\n. o o", side=Side.LEFT, size=3))

        # set right side objects
        self.visual_objects.append(VisualObject(". . .\n. . .\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . o\n. . .\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject("o . .\n. . .\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\n. . o\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\no . .\n. . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\n. . .\n. . o", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\n. . .\no . .", side=Side.BOTH, size=3))
        self.visual_objects.append(VisualObject(". . .\no o .\no o .", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject("o o .\no o .\n. . .", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject("o o .\no o .\no o .", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject("o . .\no . .\no o o", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject("o o o\no . .\no . .", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject("o . .\no o .\no o o", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject("o o o\no o .\no . .", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject(". o o\n. . o\n. . .", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject(". . .\n. . o\n. o o", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject("o o .\no o o\no o o", side=Side.RIGHT, size=3))
        self.visual_objects.append(VisualObject("o o o\no o o\no o .", side=Side.RIGHT, size=3))


    def get_novelty_characteristic(self, neural_network):

        # behavior vector
        behavior = []

        # test patterns to feed through the network to get behavior
        test_patterns = [(VisualObject("o o o\no o o\no o o", side=Side.LEFT, size=3),
                          VisualObject("o o o\no o o\no o o", side=Side.RIGHT, size=3)),
                         (VisualObject(". . .\no o o\no o o", side=Side.LEFT, size=3),
                          VisualObject("o o o\no o o\n. . .", side=Side.RIGHT, size=3)),
                         (VisualObject("o o o\n. . .\n. . .", side=Side.LEFT, size=3),
                          VisualObject(". . .\n. . .\no o o", side=Side.RIGHT, size=3)),
                         (VisualObject("o . .\no . .\no . .", side=Side.LEFT, size=3),
                          VisualObject(". . o\n. . o\n. . o", side=Side.RIGHT, size=3))]

        # iterate test patterns
        for test_pattern_left, test_pattern_right in test_patterns:

            test_input = test_pattern_left.get_data() + test_pattern_right.get_data()

            network_output, _ = neural_network.activate(test_input)
            behavior.extend(network_output)

        return behavior