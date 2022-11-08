import os
import multiprocessing

import neat
import visualize
from neat.nsga2 import NSGA2Reproduction
from util import clear_checkpoints, load_checkpoints, make_new_run_folders


class Neuroevolution:

    def __init__(self, domain, simulator, objectives, config_file, num_generations, evaluator=None):
        self.domain = domain
        self.objectives = objectives
        self.simulator = simulator(self.objectives)
        self.config_file = config_file
        self.num_generations = num_generations
        self.evaluator = evaluator

        reproduction = neat.DefaultReproduction
        if len(objectives) > 1:
            reproduction = NSGA2Reproduction

        self.neat_config = neat.Config(neat.DefaultGenome, reproduction, 
                                       neat.DefaultSpeciesSet, neat.NoStagnation,
                                       self.config_file)

        self.pop = neat.Population(self.neat_config)

        self.results_data_path, self.results_plot_path, self.checkpoint_path = make_new_run_folders(self.domain)
        self.init_reporters_and_checkpoints()

    def init_reporters_and_checkpoints(self):

        self.stats = neat.StatisticsReporter()
        self.pop.add_reporter(self.stats)
        self.pop.add_reporter(neat.StdOutReporter(False))
        self.pop.add_reporter(neat.Checkpointer(generation_interval=5,
                                                filename_prefix=os.path.join(self.checkpoint_path, "gen_")))

    def run(self):
        if self.evaluator is not None:
            evaluator = self.evaluator(multiprocessing.cpu_count(), self.simulator)
            best_genome = self.pop.run(evaluator.evaluate, n=self.num_generations)
        else:
            best_genome = self.pop.run(self.simulator.evaluate_genomes, n=self.num_generations)

        return best_genome

    def save_stats(self):

        # save record store data
        if self.domain == "mazerobot":
            record_store_save_file = os.path.join(self.results_data_path, "agent_record_data.pickle")
            self.simulator.history.dump(record_store_save_file)

        # save fitness
        fitness_save_file = os.path.join(self.results_data_path, "best_and_mean_fitness_data.csv")
        self.stats.save_genome_fitness(delimiter=",", filename=fitness_save_file)

    def visualize_stats(self, winner_genome):
        labels = {"performance":"Task Performance",
                  "hamming":"Hamming Distance",
                  "beh_div":"Behavioural Diversity"}

        print('\nBest genome:\n{!s}'.format(winner_genome))
        winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, self.neat_config)

        # draw network of best genome
        net_save_file = os.path.join(self.results_plot_path, f"{self.domain}_net")
        visualize.draw_net(winner_net, net_save_file)

        # plot stats
        plot_stats_save_file = os.path.join(self.results_plot_path, "avg_fitness.png")
        visualize.plot_stats(self.stats, plot_stats_save_file, ylog=False, view=True)

        # plot pareto front
        checkpoints = load_checkpoints(self.checkpoint_path)
        plot_pareto_front_file = os.path.join(self.results_plot_path, f"pareto_front.png")
        if self.domain == "xor":
            visualize.plot_pareto_2d(checkpoints, plot_pareto_front_file, "XOR",
                                     labels[self.objectives[0]], labels[self.objectives[1]],
                                     4.0, 10.0)
        elif self.domain == "retina":
            visualize.plot_pareto_2d(checkpoints, plot_pareto_front_file, "Retina",
                                     labels[self.objectives[0]], labels[self.objectives[1]],
                                     1.0, 500.0)
        elif self.domain == "mazerobot":
            visualize.plot_pareto_2d(checkpoints, plot_pareto_front_file, "Mazerobot",
                                     labels[self.objectives[0]], labels[self.objectives[1]],
                                     13.5, 500.0)
        elif self.domain == "bipedal":
            visualize.plot_pareto_2d(checkpoints, plot_pareto_front_file, "BipedalWalker",
                                     labels[self.objectives[0]], labels[self.objectives[1]],
                                     500.0, 10000.0)
