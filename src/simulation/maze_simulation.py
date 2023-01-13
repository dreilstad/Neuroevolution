import os
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

    def simulate(self, neural_network):

        all_activations = None
        if self.CKA is not None:
            all_activations = []

        sequence = None
        if self.hamming is not None:
            sequence = []

        exit_found = False

        for i in range(self.MAX_TIME_STEPS):

            # activate
            state = self.env.create_net_inputs()
            output, activations = neural_network.activate(state)

            # step
            exit_found = self.env.update(output)

            # save sequence if using hamming distance
            if self.hamming is not None:
                sequence.extend([*state, *output])

            # append activations if using CKA
            if self.CKA is not None:
                all_activations.append(activations)

            if exit_found:
                break

        novelty = self._get_novelty_characteristic(None)

        # Calculate the fitness score based on distance from exit
        task_performance = 1.0
        if not exit_found:
            # Normalize distance to range (0,1]
            distance_to_exit = self.env.agent_distance_to_exit()
            task_performance = (self.env.initial_distance - distance_to_exit) / self.env.initial_distance

        return [task_performance, self._binarize_sequence(sequence), novelty, all_activations]

    def simulate(self, genome_id, neural_network):

        task_performance, sequence, all_activations, novelty = self.maze_simulation_evaluate(
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

        # [performance, hamming, novelty, CKA, Q]
        return [task_performance, self._binarize_sequence(sequence), novelty, all_activations]

    def _get_novelty_characteristic(self, neural_network):
        return [self.env.agent.location.x, self.env.agent.location.y]