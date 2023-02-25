import os
import multiprocessing as mp
import neat
import visualize
from neat.nsga2 import NSGA2Reproduction
from util import clear_checkpoints, load_checkpoints, make_new_run_folders


class Neuroevolution:

    def __init__(self, domain, simulator, objectives, config_file, num_generations, show, evaluator=None):
        self.domain = domain
        self.objectives = objectives
        self.simulator = simulator(self.objectives, self.domain)
        self.config_file = config_file
        self.num_generations = num_generations
        self.show = show
        self.evaluator = evaluator

        self.neat_config = neat.Config(neat.DefaultGenome, NSGA2Reproduction, self.config_file)

        self.pop = neat.Population(self.neat_config)

    def init_reporters_and_checkpoints(self):

        self.stats = neat.StatisticsReporter()
        self.pop.add_reporter(self.stats)
        self.pop.add_reporter(neat.StdOutReporter(False))
        self.pop.add_reporter(neat.Checkpointer(generation_interval=5,
                                                filename_prefix=os.path.join(self.checkpoint_path, "gen_")))

    def run(self):
        self.results_data_path, self.results_plot_path, self.checkpoint_path = make_new_run_folders(self.domain,
                                                                                                    self.objectives)
        self.init_reporters_and_checkpoints()
        """
        g = self.pop.population[1]
        net = neat.nn.FeedForwardNetwork.create(g, self.neat_config)
        visualize.draw_net(net, f"init_net")
        for i in range(21):
            g.mutate(self.neat_config.genome_config)
            if i % 10 == 0:
                net = neat.nn.FeedForwardNetwork.create(g, self.neat_config)
                visualize.draw_net(net, f"{i}_net")

        exit(0)
        """

        if self.evaluator is not None:
            evaluator = self.evaluator(40, self.simulator)
            best_genome = self.pop.run(evaluator.evaluate_genomes, n=self.num_generations)
        else:
            best_genome = self.pop.run(self.simulator.evaluate_genomes, n=self.num_generations)

        return best_genome

    def save_stats(self):

        # save record store data
        if self.domain == "mazerobot-medium" or self.domain == "mazerobot-hard":
            record_store_save_file = os.path.join(self.results_data_path, "agent_record_data.pickle")
            self.simulator.history.dump(record_store_save_file)
            if "beh_div" in self.objectives:
                archive_save_file = os.path.join(self.results_data_path, "archive_data.pickle")
                self.simulator.novelty.write_archive_to_file(archive_save_file)

        # save fitness
        self.save_genome_fitness(self.results_data_path)

    def visualize_stats(self, winner_genome):
        labels = {"performance": "Task Performance",
                  "hamming": "Hamming Distance",
                  "beh_div": "Behavioural Diversity (ad hoc)",
                  "modularity": "Modularity (Q-score)",
                  "mod_div": "Modularity Diversity",
                  "cos_sim": "Cosine Similarity",
                  "linear_cka": "Linear CKA",
                  "rbf_cka": "RBF CKA"}

        domain_labels = {"retina": "Retina 2x2",
                         "retina-hard": "Retina 3x3",
                         "bipedal": "Bipedal Walker",
                         "tartarus": "Tartarus",
                         "tartarus-deceptive": "Deceptive Tartarus",
                         "mazerobot-medium": "Maze navigation - Medium maze",
                         "mazerobot-hard": "Maze navigation - Hard maze"}

        print('\nBest genome:\n{!s}'.format(winner_genome))
        winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, self.neat_config)

        # draw network of best genome
        net_save_file = os.path.join(self.results_plot_path, f"{self.domain}_net")
        visualize.draw_net(winner_net, net_save_file)

        # plot stats
        plot_stats_save_file = os.path.join(self.results_plot_path, "avg_fitness.png")
        visualize.plot_stats(self.stats, plot_stats_save_file, ylog=False, show=self.show)

        # plot pareto front
        if len(self.objectives) > 1:
            checkpoints = load_checkpoints(self.checkpoint_path)
            plot_pareto_front_file = os.path.join(self.results_plot_path, f"pareto_front.pdf")

            visualize.plot_pareto_2d_fronts(checkpoints, plot_pareto_front_file, domain_labels[self.domain],
                                            labels[self.objectives[0]], labels[self.objectives[1]])

        clear_checkpoints(self.checkpoint_path, save_last=True)

    def save_genome_fitness(self, save_path):
        run_number = os.path.basename(os.path.normpath(save_path))

        generation = range(self.num_generations)
        best_fitness = [c.fitness.values[0] for c in self.stats.most_fit_genomes]

        # check if run terminated early, fill in last fitness value to match num_generations
        if len(best_fitness) < self.num_generations:
            best_fitness = best_fitness + [best_fitness[-1]] * (self.num_generations - len(best_fitness))

        with open(os.path.join(self.results_data_path, f"best_fitness_{run_number}.dat"), "w") as f:
            for gen, fitness in zip(generation, best_fitness):
                f.write(f"{gen} {fitness}\n")
