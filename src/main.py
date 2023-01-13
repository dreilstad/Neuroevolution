import argparse
import neat
from neuroevolution import Neuroevolution
from util import validate_input


def main(domain, simulator, objectives, config_file, num_generations, show, parallel):
    evaluator = None
    if parallel:
        evaluator = neat.MultiObjectiveParallelEvaluator

    num_experiments = 25
    for _ in range(num_experiments):
        print(type(simulator))
        ne = Neuroevolution(domain, simulator, objectives, config_file,
                            num_generations, show, evaluator=evaluator)
        winner = ne.run()
        ne.visualize_stats(winner)
        ne.save_stats()

if __name__=="__main__":

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

    print(args.domain)
    print(args.config_file)
    print(args.objectives)
    print(args.show)
    print(args.parallel)
    simulator, config, objectives, num_generations = validate_input(args)

    main(args.domain, simulator, objectives, config, num_generations, args.show, args.parallel)
    """

    from simulation.environments.tartarus.tartarus_environment import TartarusEnvironment
    from simulation.environments.tartarus.minigrid.manual_control import ManualControl
    from simulation.environments.tartarus.tartarus_util import generate_configurations
    import numpy as np

    N_size = 6
    K_boxes = 6
    configs = generate_configurations(N_size, K_boxes)
    env = TartarusEnvironment(configs[np.random.randint(len(configs))])
    env.reset()
    env.agent_view_size = 3

    print(env.encode_tartarus_state())
    print(env.encode_tartarus_state_with_walls())
    print(f"performance score: {env.state_evaluation()}")

    manual_control = ManualControl(env=env, agent_view=False)
    manual_control.start()

    print(f"performance score: {env.state_evaluation()}")
    print(env.encode_tartarus_state())
    print(env.encode_tartarus_state_with_walls())
    """
