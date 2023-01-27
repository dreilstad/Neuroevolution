import networkx as nx
from networkx.algorithms import community
from collections import Counter

class Modularity:

    @staticmethod
    def calculate_modularity(nodes, edges):
        graph = Modularity.network_to_graph(nodes, edges)
        communities = community.greedy_modularity_communities(graph)
        return community.modularity(graph, communities)

    @staticmethod
    def network_to_graph(nodes, edges):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph


class ModularityDiversity:

    def __init__(self):
        self.mod_div = {}
        self.decompositions = {}

    def __getitem__(self, key):
        return self.mod_div[key]

    def add(self, genome_id, nodes, edges, nodes_of_interest):
        graph = self.network_to_graph(nodes, edges)
        communities = community.greedy_modularity_communities(graph)
        communities_without_excess = self._remove_excess_nodes(communities, nodes_of_interest)

        self.decompositions[genome_id] = communities_without_excess

    def calculate_modular_diversity(self, genomes):

        # reset ModDiv scores dict
        self.mod_div = {}
        number_of_comparisons = float(len(genomes) - 1)

        for genome_id, _ in genomes:
            modular_diversity = 0.0
            for other_id, _ in genomes:
                if genome_id != other_id:
                    M_evo = self.decompositions[genome_id]
                    M_comp = self.decompositions[other_id]
                    modular_diversity += self.modular_diversity(M_evo, M_comp)

            self.mod_div[genome_id] = modular_diversity / number_of_comparisons

        # reset decompositions dict
        self.decompositions = {}

    def modular_diversity(self, M_evo, M_comp):
        uniformity = self.uniformity(M_evo, M_comp)
        conflicts = self.conflicts(M_evo, M_comp)
        return 1.0 - ((uniformity + conflicts) / 2.0)

    def uniformity(self, M_evo, M_comp):
        total_number_neurons = sum(len(module) for module in M_comp)
        uniform_neurons_counter = 0

        for module in M_comp:
            evolved_neuron_colors = self._get_colors(M_evo, module)
            main_color, module_uniformity = self._most_common_color_and_num_occurrences(evolved_neuron_colors)
            uniform_neurons_counter += module_uniformity

        return uniform_neurons_counter / total_number_neurons

    def conflicts(self, M_evo, M_comp):
        num_conflicts = 0
        max_num_conflicts = 0

        for i, module_A in enumerate(M_comp):
            module_A_colors = self._get_colors(M_evo, module_A)
            other_comp_modules = M_comp[:i] + M_comp[i+1:]
            for module_B in other_comp_modules:
                module_B_colors = self._get_colors(M_evo, module_B)
                num_conflicts += self._count_matches(module_A_colors, module_B_colors)
                max_num_conflicts += len(module_A) * len(module_B)

        if max_num_conflicts == 0:
            return 0.0

        return (max_num_conflicts - num_conflicts) / max_num_conflicts

    @staticmethod
    def _remove_excess_nodes(communities, nodes_of_interest):
        communities_without_excess = []
        for module in communities:
            module_without_excess = []
            for node in module:
                if node in nodes_of_interest:
                    module_without_excess.append(node)

            if len(module_without_excess) > 0:
                communities_without_excess.append(module_without_excess)

        return communities_without_excess

    @staticmethod
    def _get_colors(M, neuron_indices):
        colors = []
        for ind in neuron_indices:
            for color, module in enumerate(M):
                if ind in module:
                    colors.append(color)
                    break

        return colors

    @staticmethod
    def _most_common_color_and_num_occurrences(evolved_neuron_colors):
        counter = Counter(evolved_neuron_colors)
        main_color, number_of_occurrences = counter.most_common(1)[0]

        return main_color, number_of_occurrences

    @staticmethod
    def _count_matches(colors, other_colors):
        matches = 0
        for color in colors:
            for other_color in other_colors:
                if color == other_color:
                    matches += 1

        return matches

    @staticmethod
    def network_to_graph(nodes, edges):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph
