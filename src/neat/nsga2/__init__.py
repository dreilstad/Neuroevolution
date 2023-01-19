"""

    Implementation of NSGA-II as a reproduction method for NEAT.
    More details on the README.md file.

    @autor: Hugo Aboud (@hugoaboud)

"""
from __future__ import division

import numpy as np
from itertools import count
from operator import add

from neat.config import ConfigParameter, DefaultClassConfig

##
#   NSGA-II Fitness
#   Stores multiple fitness values
#   Overloads operators allowing integration to unmodified neat-python
##

class NSGA2Fitness:
    def __init__(self, *values):
        self.values = values
        self.rank = 0
        self.dist = 0.0

    def set(self, *values):
        self.values = values

    def add(self, *values):
        self.values = list(map(add, self.values, values))

    def dominates(self, other):
        d = False
        for a, b in zip(self.values, other.values):
            if (a < b): return False
            elif (a > b): d = True
        return d

    # >
    def __gt__(self, other):
        # comparison of fitnesses on tournament, use crowded-comparison operator
        # this is also used by max/min
        if isinstance(other, NSGA2Fitness):
            if self.rank > other.rank:
                return True
            elif self.rank == other.rank and self.dist > other.dist:
                return True

            return False
        # stagnation.py initializes fitness as -sys.float_info.max
        # it's the only place where the next line should be called
        return self.rank > other

    # >=
    def __ge__(self, other):
        # population.run() compares fitness to the fitness threshold for termination
        # it's the only place where the next line should be called
        # it's also the only place where score participates of evolution
        # besides that, score is a value for reporting the general evolution
        return self.values[0] >= other

    # -
    def __sub__(self, other):
        # used only by reporting->neat.math_util to calculate fitness (score) variance
        return self.values[0] - other

    # float()
    def __float__(self):
        # used only by reporting->neat.math_util to calculate mean fitness (score)
        return float(self.values[0])

    # str()
    def __str__(self):
        return "rank:{0},dist:{1},values:{2}".format(self.rank, self.dist, self.values)

##
#   NSGA-II Reproduction
#   Implements "Non-Dominated Sorting" and "Crowding Distance Sorting" to reproduce the population
##

class NSGA2Reproduction(DefaultClassConfig):
    @classmethod
    def parse_config(cls, param_dict):

        return DefaultClassConfig(param_dict,
                                  [ConfigParameter("tournament", str, "random_selection")])

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)

        # Parent population and species
        # This population is mixed with the evaluated population in order to achieve elitism
        self.parent_pop = []
        self.parent_species = {}

        # Parento-fronts of genomes (including population and parent population)
        # These are created by the sort() method at the end of the fitness evaluation process
        self.fronts = []

        # tournament dict
        self.tournament_type = {"random_selection": self.random_selection,
                                "tournament_selection": self.tournament_selection}

    # new population, called by the population constructor
    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
        return new_genomes

    # NSGA-II step 1: fast non-dominated sorting
    # This >must< be called by the fitness function (aka eval_genomes)
    # after a NSGA2Fitness was assigned to each genome
    def sort(self, population, pop_size):

        # NSGA-II : step 1 : merge and sort
        # Merge populations P(t)+Q(t) and sort by non-dominated fronts
        child_pop = [g for _, g in population.items()] + self.parent_pop

        # Non-Dominated Sorting (of P(t)+Q(t))
        # algorithm data
        S = {} # genomes dominated by key genome
        n = {} # counter of genomes dominating key genome
        F = [] # current dominance front
        self.fronts = [] # clear dominance fronts
        # calculate dominance of every genome to every other genome - O(MN²)
        for p in range(len(child_pop)):
            S[p] = []
            n[p] = 0
            for q in range(len(child_pop)):
                if p == q:
                    continue
                # p dominates q
                if child_pop[p].fitness.dominates(child_pop[q].fitness):
                    S[p].append(q)
                # q dominates p
                elif child_pop[q].fitness.dominates(child_pop[p].fitness):
                    n[p] += 1
            # if genome is non-dominated, set rank and add to front
            if n[p] == 0:
                child_pop[p].fitness.rank = 0
                F.append(p)

        # assemble dominance fronts - O(N²)
        i = 0 # dominance front iterator
        while len(F) > 0:
            # store front
            self.fronts.append([child_pop[f] for f in F])
            # new dominance front
            Q = []
            # for each genome in current front
            for p in F:
                # for each genome q dominated by p
                for q in S[p]:
                    # decrease dominate counter of q
                    n[q] -= 1
                    # if q reached new front
                    if n[q] == 0:
                        child_pop[q].fitness.rank = -(i+1)
                        Q.append(q)
            # iterate front
            i += 1
            F = Q

        # NSGA-II : step 2 : pareto selection
        # Create new parent population P(t+1) from the best fronts
        # Sort each front by Crowding Distance, to be used on Tournament
        self.parent_pop = []
        for front in self.fronts:
            # Calculate crowd-distance of fitnesses
            # First set distance to zero
            for genome in front:
                genome.dist = 0
            # List of fitnesses to be used for distance calculation
            fitnesses = [f.fitness for f in front]
            # Iterate each fitness parameter (values)
            for m in range(len(fitnesses[0].values)):
                # Sort fitnesses by parameter
                fitnesses.sort(key=lambda f: f.values[m])
                # Get scale for normalizing values
                scale = (fitnesses[-1].values[m]-fitnesses[0].values[m])
                # Set edges distance to infinite, to ensure are picked by the next step
                # This helps keeping the population diverse
                fitnesses[0].dist = float('inf')
                fitnesses[-1].dist = float('inf')
                # Increment distance values for each fitness
                if scale > 0:
                    for i in range(1,len(fitnesses)-1):
                        fitnesses[i].dist += abs(fitnesses[i+1].values[0]-fitnesses[i-1].values[0])/scale

            # Sort front by crowd distance
            # In case distances are equal (mostly on 'inf' values), use the first value to sort
            front.sort(key=lambda g: (g.fitness.dist, g.fitness.values[0]), reverse=True)

            # Assemble new parent population P(t+1)
            # front fits entirely on the parent population, just append it
            if (len(self.parent_pop) + len(front) <= pop_size):
                self.parent_pop += front
                if (len(self.parent_pop) == pop_size): break
            # front exceeds parent population, append only what's necessary to reach pop_size
            else:
                self.parent_pop += front[:pop_size-len(self.parent_pop)]
                break

        # NSGA-II : end : return parent population P(t+1) to be assigned to child population container Q(t+1)
        # this container will be used on the Tournament at NSGA2Reproduction.reproduce()
        # to create the real Q(t+1) population
        return {g.key:g for g in self.parent_pop}

    # NSGA-II step 2: crowding distance sorting
    # this is where NSGA-2 reproduces the population by the fitness rank
    # calculated on step 1
    def reproduce(self, config, population):

        ## NSGA-II : step 3 : Tournament
        # Disclaimer: this method uses no absolute fitness values
        # The fitnesses are compared through the crowded-comparison operator
        # fitness.values[0] is used for fitness threshold and reporting, but not in here

        ## Tournament
        # Each species remains the same size (they grow and shrink based on pareto-fronts, on sort())
        # Only the <survival_threshold> best are used for mating
        # Mating can be sexual or asexual

        selection = self.tournament_type[self.reproduction_config.tournament]
        new_population = {}
        for i in range(len(population)):
            parent = selection(list(population.items()))

            genome_id = next(self.genome_indexer)
            child = config.genome_type(genome_id)
            child.copy_genes(parent)

            child.mutate(config.genome_config)
            new_population[genome_id] = child

        return new_population

    @staticmethod
    def random_selection(population):
        selection_ind = np.random.choice(len(population))
        selection = population[selection_ind]
        return selection[1]

    @staticmethod
    def tournament_selection(population, fraction=0.2):

        # select 20% of the population, and sort by fitness
        selection_indices = np.random.choice(len(population), size=int(len(population) * fraction), replace=False)
        selection = [population[i] for i in selection_indices]
        selection.sort(key=lambda g: g[1].fitness, reverse=True)

        # return winner of selection
        return selection[0][1]
