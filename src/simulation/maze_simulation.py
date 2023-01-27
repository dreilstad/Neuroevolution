import os
from simulation.simulator import Simulator
from simulation.environments.maze.maze_environment import read_environment
from simulation.environments.maze.agent import AgentRecordStore, AgentRecord


class MazeSimulator(Simulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        self.history = AgentRecordStore()
        self.MAX_TIME_STEPS = 400
        self.use_input_nodes_in_mod_div = True

    def simulate(self, neural_network):

        self.env.reset()

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

        # create agent record
        record = AgentRecord()
        record.distance = task_performance
        record.x = self.env.agent.location.x
        record.y = self.env.agent.location.y
        record.hit_exit = exit_found

        return {"performance": task_performance,
                "hamming": self._binarize_sequence(sequence),
                "novelty": novelty,
                "CKA": all_activations,
                "agent_record": record}

    def _get_novelty_characteristic(self, neural_network):
        return [self.env.agent.location.x, self.env.agent.location.y]


class MediumMazeSimulator(MazeSimulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        local_dir = os.path.dirname(os.path.realpath(__file__))
        maze_config_dir = os.path.join(local_dir, "environments/maze/medium_maze.txt")
        self.env = read_environment(maze_config_dir)


class HardMazeSimulator(MazeSimulator):

    def __init__(self, objectives, domain):
        super().__init__(objectives, domain)
        local_dir = os.path.dirname(os.path.realpath(__file__))
        maze_config_dir = os.path.join(local_dir, "environments/maze/hard_maze.txt")
        self.env = read_environment(maze_config_dir)
