import argparse
import neat
import os
from neuroevolution import Neuroevolution
from util import validate_input


def main(domain, simulator, objectives, config_file, num_generations, show, parallel):
    evaluator = None
    if parallel:
        evaluator = neat.MultiObjectiveParallelEvaluator

    num_experiments = 10
    for _ in range(num_experiments):
        ne = Neuroevolution(domain, simulator, objectives, config_file,
                            num_generations, show, evaluator=evaluator)
        winner = ne.run()
        ne.visualize_stats(winner)
        ne.save_stats()

if __name__=="__main__":

    # TODO: ved kjøring på ml nodes, bruk:
    #       module load PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised


    parser = argparse.ArgumentParser(description="Running Neuroevolution experiment",
                                     add_help=True)
    parser.add_argument("-d", "--domain", 
                        type=str,
                        required=True,
                        help="Name of domain/task/problem to simulate")

    parser.add_argument("-c", "--config_file",
                        type=str,
                        required=True,
                        help="Name of config file to be used")

    parser.add_argument("-n", "--num_generations",
                        type=int,
                        required=True,
                        help="Number of generations to evolve")
                    
    parser.add_argument("-o", "--objectives",
                        nargs="+",
                        required=True,
                        help="List of objectives to use")

    parser.add_argument("-s", "--show",
                        action="store_true",
                        help="Shows plots if flag is present")

    parser.add_argument("-p", "--parallel",
                        action="store_true",
                        help="Runs experiments in parallel if flag is present")

    args = parser.parse_args()

    print(f"Domain: {args.domain}")
    print(f"Config: {args.config_file}")
    print(f"Objectives: {args.objectives}")
    print(f"Parallel: {args.parallel}")
    print(f"Show figures: {args.show}")
    simulator, config, objectives, num_generations = validate_input(args)

    main(args.domain, simulator, objectives, config, num_generations, args.show, args.parallel)
    """
    from simulation.environments.tartarus.tartarus_environment import TartarusEnvironment
    from simulation.environments.tartarus.minigrid.manual_control import ManualControl
    from simulation.environments.tartarus.tartarus_util import generate_configurations
    import numpy as np

    N_size = 6
    K_boxes = 6
    configs, test_config, pos, direction = generate_configurations(N_size, K_boxes, sample=10)
    #config = configs[np.random.randint(len(configs))]
    print(test_config)

    env = TartarusEnvironment(test_config)
    env.reset()
    env.agent_view_size = 3

    print(env.encode_tartarus_state())
    print(env.encode_tartarus_state_with_walls())
    print(f"performance score: {env.state_evaluation()}")

    #print(env.get_initial_block_positions())
    manual_control = ManualControl(env=env, agent_view=False)
    manual_control.start()

    print(f"performance score: {env.state_evaluation()}")
    print(env.encode_tartarus_state())
    print(env.encode_tartarus_state_with_walls())

    #print(env.get_initial_block_positions())

    from simulation.environments.maze.agent import AgentRecordStore
    from simulation.environments.maze.maze_environment import read_environment
    from simulation.environments.maze.visualize import draw_maze_records
    import random
    import os


    parser = argparse.ArgumentParser(description="The maze experiment visualizer.")
    parser.add_argument('-m', '--maze', default='medium', help='The maze configuration to use.')
    parser.add_argument('-r', '--records', help='The records file.')
    parser.add_argument('-o', '--output', help='The file to store the plot.')
    parser.add_argument('--width', type=int, default=300, help='The width of the subplot')
    parser.add_argument('--height', type=int, default=140, help='The height of the subplot')
    parser.add_argument('--fig_height', type=float, default=7, help='The height of the plot figure')
    parser.add_argument('--show_axes', type=bool, default=False, help='The flag to indicate whether to show plot axes.')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    if not (args.maze == 'medium' or args.maze == 'hard'):
        print('Unsupported maze configuration: %s' % args.maze)
        exit(1)

    # read maze environment
    maze_env_config = os.path.join(local_dir, '%s_maze.txt' % args.maze)
    import time

    maze_env = read_environment("/Users/didrik/Documents/Master/Neuroevolution/src/simulation/environments/maze/medium_maze.txt")

    # read agents records
    rs = AgentRecordStore()
    rs.load(args.records)

    # render visualization
    draw_maze_records(maze_env,
                      rs.records,
                      width=args.width,
                      height=args.height,
                      fig_height=args.fig_height,
                      show_axes=args.show_axes,
                      filename=args.output)
    """
