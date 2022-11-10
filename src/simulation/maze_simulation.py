import os
import copy
from scipy.spatial import distance
from simulation.simulator import Simulator
from simulation.environments.maze.maze_environment import read_environment
from simulation.environments.maze.agent import AgentRecordStore, AgentRecord
from simulation.environments.maze.geometry import Point


class MazeSimulator(Simulator):

    def __init__(self, objectives, maze_config="medium_maze.txt"):
        self.metric = self.novelty_metric
        super().__init__(objectives)

        local_dir = os.path.dirname(os.path.realpath(__file__))
        maze_config_dir = os.path.join(local_dir, "environments/maze/")
        self.env = read_environment(maze_config_dir + maze_config)
        self.history = AgentRecordStore()

        self.MAX_TIME_STEPS = 400

    def simulate(self, genome_id, genome, neural_network, generation):

        maze = copy.deepcopy(self.env)
        task_performance, sequence, all_activations = self.maze_simulation_evaluate(env=maze,
                                                                                    genome=genome,
                                                                                    net=neural_network,
                                                                                    time_steps=self.MAX_TIME_STEPS)

        record = AgentRecord(generation=generation, agent_id=genome_id)
        record.distance = task_performance
        record.x = maze.agent.location.x
        record.y = maze.agent.location.y
        record.hit_exit = maze.exit_found
        record.species_id = 1
        self.history.add_record(record)

        # [performance, hamming, Q, CKA]
        return [task_performance, self._binarize_sequence(sequence), None, all_activations]

    def maze_simulation_evaluate(self, env, genome, net, time_steps, path_points=None):
        """
        The function to evaluate maze simulation for specific environment
        and control ANN provided. The results will be saved into provided
        agent record holder.
        Arguments:
            env:            The maze configuration environment.
            genome:         The genome of individual.
            net:            The maze solver agent's control ANN.
            time_steps:     The number of time steps for maze simulation.
            path_points:    The holder for path points collected during simulation. If
                            provided None then nothing will be collected.
        Returns:
            The goal-oriented fitness value, i.e., how close is agent to the exit at
            the end of simulation.
        """

        all_activations = None
        if self.CKA is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        exit_found = False
        for i in range(time_steps):
            network_inputs = env.create_net_inputs()
            network_output, activations = net.activate(network_inputs)
            exit_found = env.update(network_output)

            # save sequence if using hamming distance
            if self.hamming is not None:
                bin_sequence = self._binarize_sequence([*network_inputs, *network_output])
                sequence.extend(bin_sequence)

            # append activations if using CKA
            if self.CKA is not None:
                all_activations.append(activations)

            if exit_found:
                print("Maze solved in %d steps" % (i + 1))
                break

            if path_points is not None:
                # collect current position
                path_points.append(Point(env.agent.location.x, env.agent.location.y))

        # store final agent coordinates as genome's novelty characteristics
        genome.behavior = [env.agent.location.x, env.agent.location.y]

        # Calculate the fitness score based on distance from exit
        fitness = 0.0
        if exit_found:
            fitness = 1.0
        else:
            # Normalize distance to range (0,1]
            distance_to_exit = env.agent_distance_to_exit()
            fitness = (env.initial_distance - distance_to_exit) / env.initial_distance

        return fitness, sequence, all_activations

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

    @staticmethod
    def novelty_metric_euclidean_distance(first_item, second_item):
        if not hasattr(first_item, "data") or not hasattr(second_item, "data"):
            return NotImplemented

        if len(first_item.data) != len(second_item.data):
            return 0.0

        return distance.euclidean(first_item.data, second_item.data)
