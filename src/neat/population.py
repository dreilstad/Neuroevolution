"""Implements the core evolution algorithm."""
from __future__ import print_function

from neat.math_util import mean
from neat.reporting import ReporterSet, StdOutReporter


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.generation = 0
            self.best_genome = None
        else:
            self.population, self.best_genome, self.generation = initial_state

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def get_generation_runtimes(self):
        for r in self.reporters.reporters:
            if isinstance(r, StdOutReporter):
                return r.generation_times

        return None

    def run(self, fitness_function, n=None):
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(self.population.items()), self.config, self.generation)

            # Call sorting method of NSGA2Reproduction
            # This is the only modification made to the main code, so the best
            # genome is evaluated before tournament, to ensure elitism
            if callable(getattr(self.reproduction, 'sort', None)):
                self.population = self.reproduction.sort(self.population, self.config.pop_size)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.fitness.values[0] > best.fitness.values[0]:
                    best = g

            self.reporters.post_evaluate(self.config, self.population, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness.values[0] > self.best_genome.fitness.values[0]:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.population, self.best_genome)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.population)

            # Report end of generation statistics
            self.reporters.end_generation(self.config, self.population)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.population, self.best_genome)

        return self.best_genome
