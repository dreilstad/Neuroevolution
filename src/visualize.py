import warnings
import graphviz
import matplotlib.pyplot as plt
import numpy as np
#import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
import neat
from neat.nsga2 import NSGA2Reproduction
def plot_stats(statistics, filename, ylog=False, show=False):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())


    plt.plot(generation, avg_fitness, 'b-', label="average")
    # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename, format="png", dpi=300)
    if show:
        plt.show()

    plt.close()


def plot_pareto_2d(checkpoints, filename, domain, label0, label1, max0, max1, invert=True, show=False):
    fitnesses = [[f.fitness for _, f in c.population.items()] for c in checkpoints]

    fig, ax = plt.subplots(figsize = (10,5))
    ax.set_title(domain + " - Solution Space")

    if invert:
        ax.set_xlabel(label1)
        ax.set_ylabel(label0)
    else:
        ax.set_xlabel(label0)
        ax.set_ylabel(label1)


    non_dom_solutions_x = []
    non_dom_solutions_y = []
    dom_solutions_x = []
    dom_solutions_y = []
    for gen in fitnesses[:-1]:
        for f in gen:
            dom_solutions_x.append(f.values[0] if (f.values[0] < max0) else max0)
            dom_solutions_y.append(f.values[1] if (f.values[1] < max1) else max1)

    for f in fitnesses[-1]:
        if f.rank == 0:
            non_dom_solutions_x.append(f.values[0] if (f.values[0] < max0) else max0)
            non_dom_solutions_y.append(f.values[1] if (f.values[1] < max1) else max1)
        else:
            dom_solutions_x.append(f.values[0] if (f.values[0] < max0) else max0)
            dom_solutions_y.append(f.values[1] if (f.values[1] < max1) else max1)

    if invert:
        ax.scatter(dom_solutions_y, dom_solutions_x, s=3, c="#3369DE", label="Dominated solutions")
        ax.scatter(non_dom_solutions_y, non_dom_solutions_x, s=3, c="#DC3220", label="Non-dominated solutions")
    else:
        ax.scatter(dom_solutions_x, dom_solutions_y, s=3, c="#3369DE", label="Dominated solutions")
        ax.scatter(non_dom_solutions_x, non_dom_solutions_y, s=3, c="#DC3220", label="Non-dominated solutions")

    sorted_x_y = [(x,y) for x, y in sorted(zip(non_dom_solutions_x, non_dom_solutions_y), key=lambda pair: pair[0])]
    x_sorted = [x for x, _ in sorted_x_y]
    y_sorted = [y for _, y in sorted_x_y]
    ax.plot(y_sorted, x_sorted, linewidth=0.5, c="#000000", label="Pareto front")

    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, format="png", dpi=300)
    if show:
        plt.show()

    plt.close()

def draw_net(net, filename, node_names={}, node_colors={}):
    """
    Draw neural network with arbitrary topology.
    """
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph('svg',
                           graph_attr={'nodesep': '0.1',
                                       'ranksep': '0.75'},
                           node_attr=node_attrs)

    with dot.subgraph() as s:
        s.attr(rank="same")
        for k in sorted(net.input_nodes):
            name = node_names.get(k, str(k))
            input_attrs = {'style': 'filled',
                           'fillcolor': "#FFB000",
                           'label': '',
                           'fixedsize': 'true',
                           'width': '0.25',
                           'height': '0.25'}
            s.node(name, _attributes=input_attrs)

    with dot.subgraph() as s:
        s.attr(rank="same")
        for k in sorted(net.output_nodes):
            name = node_names.get(k, str(k))
            output_attrs = {'style': 'filled',
                            'fillcolor': "#DC267F",
                            'label': '',
                            'fixedsize': 'true',
                            'width': '0.25',
                            'height': '0.25'}
            s.node(name, _attributes=output_attrs)

    edges = []
    for node, _, _, _, _, links in net.node_evals:
        for i, _ in links:
            edges.append((i, node))

    hidden_layers = [[]]
    current_layer = [*net.input_nodes]
    next_layer = []

    i = 1
    while len(current_layer) > 0:
        for edge in edges:
            if edge[0] in current_layer:
                if edge[1] not in next_layer and edge[1] not in net.output_nodes:
                    next_layer.append(edge[1])
                    for layer in hidden_layers[:i]:
                        if edge[1] in layer:
                            layer.remove(edge[1])

        hidden_layers.append(next_layer)
        current_layer = next_layer
        next_layer = []
        i += 1

    print(filename)
    for i, layer in enumerate(hidden_layers):
        print(f"layer {i}: {layer}")
    print()

    for layer in hidden_layers:
        with dot.subgraph() as s:
            s.attr(rank="same")
            for node, _, _, _, _, links in net.node_evals:
                if node in layer and node not in net.output_nodes:
                    name = node_names.get(node, str(node))
                    hidden_attrs = {'style': 'filled',
                                    'fillcolor': "#648FFF",
                                    'label': '',
                                    'fixedsize': 'true',
                                    'width': '0.25',
                                    'height': '0.25'}
                    s.node(name, _attributes=hidden_attrs)

    input_nodes = sorted(net.input_nodes)
    for i in range(1, len(input_nodes)):
        dot.edge(str(input_nodes[i-1]), str(input_nodes[i]), _attributes={'style': 'invisible',
                                                                          'color': 'white',
                                                                          'arrowhead': 'none'})

    output_nodes = sorted(net.output_nodes)
    for i in range(1, len(output_nodes)):
        dot.edge(str(output_nodes[i-1]), str(output_nodes[i]), _attributes={'style': 'invisible',
                                                                            'color': 'white',
                                                                            'arrowhead': 'none'})

    for i, j in edges:
        a = node_names.get(i, str(i))
        b = node_names.get(j, str(j))
        style = 'solid'
        color = 'black'
        dot.edge(a, b, _attributes={
                 'style': style, 'color': color, 'dir':'forward', 'arrowhead': 'none'})

    dot.render(filename)
    return dot

if __name__=="__main__":
    from util import load_checkpoints
    check = load_checkpoints("/Users/didrik/Documents/Master/Neuroevolution/src/checkpoints/retina/performance-hamming/000")
    #save_file = "/Users/didrik/Documents/Master/Neuroevolution/src/results/plots/retina/performance-hamming/050/pareto_front_test.png"
    #plot_pareto_2d(check, save_file, "Retina",
    #                         "Task Performance", "Hamming Distance",
    #                         500.0, 10000.0)
    config_file = "/Users/didrik/Documents/Master/Neuroevolution/src/config/retina_config_multiobj.ini"
    i = 0
    #for c in check[-1]:
    for genome_id, genome in check[-1].population.items():
        if genome.fitness.rank == 0:
            network = neat.nn.FeedForwardNetwork.create(genome, neat.Config(neat.DefaultGenome, NSGA2Reproduction,
                                                                            neat.DefaultSpeciesSet, neat.NoStagnation,
                                                                            config_file))
            draw_net(network, f"{i}_network")
            """
            G = toNetworkxGraph(network)

            color_map = []
            for node in G:
                if node in network.input_nodes:
                    color_map.append("#FFB000")
                elif node in network.output_nodes:
                    color_map.append("#DC267F")
                else:
                    color_map.append("#648FFF")

            pos = graphviz_layout(G, prog="dot")

            nx.draw(G, node_color=color_map, node_size=500, with_labels=True, pos=pos, font_weight="bold")
            color_labels = {"Input nodes":"#FFB000", "Hidden nodes":"#648FFF", "Output nodes":"#DC267F"}
            for name, color in color_labels.items():
                plt.scatter([], [], c=color, label=name)

            plt.legend()
            plt.savefig(f"{i}_network.png")
            plt.close()
            """

            i += 1

    print(i)