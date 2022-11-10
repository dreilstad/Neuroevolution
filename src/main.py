import argparse
from neuroevolution import Neuroevolution
from util import validate_input, load_checkpoints
import neat


def main(domain, simulator, objectives, config_file, num_generations):
    evaluator = None
    #evaluator = neat.MultiObjectiveParallelEvaluator
    ne = Neuroevolution(domain, simulator, objectives, config_file,
                        num_generations=num_generations, evaluator=evaluator)
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

    args = parser.parse_args()

    print(args.domain)
    print(args.config_file)
    print(args.objectives)

    simulator, config, objectives, num_generations = validate_input(args)

    main(args.domain, simulator, objectives, config, num_generations)
