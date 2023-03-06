import warnings
import graphviz
import argparse

import matplotlib.pyplot as plt
import numpy as np

from simulation.environments.maze.agent import AgentRecordStore
from simulation.environments.maze.maze_environment import read_environment


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


def plot_pareto_2d_fronts(checkpoints, filename, domain, label0, label1, invert=True):
    fitnesses = [[f.fitness for _, f in c.population.items()] for c in checkpoints]
    generations = [c.generation + 1 for c in checkpoints]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(domain, fontsize=15)

    if invert:
        ax.set_xlabel(label1, fontsize=15)
        ax.set_ylabel(label0, fontsize=15)
    else:
        ax.set_xlabel(label0, fontsize=15)
        ax.set_ylabel(label1, fontsize=15)

    cmap = plt.get_cmap("plasma_r")
    norm = plt.Normalize(generations[0], generations[-1])

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    cb.set_label(label="Generation", size=15)
    cb.ax.tick_params(labelsize=15)

    fronts_x = []
    fronts_y = []
    for gen in fitnesses:
        front_x = []
        front_y = []
        for fitness in gen:
            if fitness.rank == 0:
                front_x.append(fitness.values[0])
                front_y.append(fitness.values[1])

        fronts_x.append(front_x)
        fronts_y.append(front_y)

    for front_x, front_y, generation in zip(fronts_x, fronts_y, generations):
        color = cmap(norm(generation))
        sorted_x_y = [(x,y) for x, y in sorted(zip(front_x, front_y), key=lambda pair: pair[0])]
        x_sorted = [x for x, _ in sorted_x_y]
        y_sorted = [y for _, y in sorted_x_y]
        ax.plot(y_sorted, x_sorted, linewidth=0.75, color=color)

        ax.scatter(y_sorted, x_sorted, s=15, color=color)


    #plt.legend()

    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    plt.close()

def plot_pareto_2d(checkpoints, filename, domain, label0, label1, max0, max1, invert=True, show=False):
    fitnesses = [[f.fitness for _, f in c.population.items()] for c in checkpoints]

    fig, ax = plt.subplots(figsize=(10, 5))
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


def draw_maze_records(maze_env, records, filename=None, archive=False,
                      width=140, height=300, fig_height=7):
    """
    The function to draw maze with recorded agents positions.
    Arguments:
        maze_env:       The maze environment configuration.
        records:        The records of solver agents collected during NEAT execution.
        best_threshold: The minimal fitness of maze solving agent's species to be included into the best ones.
        filename:       The name of file to store plot.
        view:           The flag to indicate whether to view plot.
        width:          The width of drawing in pixels
        height:         The height of drawing in pixels
        fig_height:      The plot figure height in inches
    """

    # initialize plotting
    fig = plt.figure()
    ax = plt.gca()

    fig_width = fig_height * (float(width) / float(height))
    fig.set_size_inches(fig_width, fig_height)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("equal")

    if records is not None:

        cmap = plt.get_cmap("gnuplot2")
        norm = plt.Normalize(records[0].generation + 1, records[-1].generation + 1)

        if not archive:
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                              ax=ax, fraction=0.02, pad=-0.00001)

            cb.set_label(label="Generation", size=20, weight="bold")
            cb.ax.tick_params(labelsize=15)

        # draw species
        _draw_species_(records=records, ax=ax, cmap=cmap, norm=norm, archive=archive)

    # draw maze
    _draw_maze_(maze_env, ax)

    # turn off axis rendering
    ax.axis('off')

    # Invert Y axis to have coordinates origin at the top left
    ax.invert_yaxis()

    # Save figure to file
    if filename is not None:
        plt.savefig(filename + ".pdf", bbox_inches="tight")

    plt.close()


def _draw_species_(records, ax, cmap, norm, archive):
    """
    The function to draw specific species from the records with
    particular color.
    Arguments:
        records:    The records of solver agents collected during NEAT execution.
        sid:        The species ID
        colors:     The colors table by species ID
        ax:         The figure axis instance
    """
    for r in records:
        if archive:
            circle = plt.Circle((r.x, r.y), 0.5, facecolor="royalblue")
        else:
            circle = plt.Circle((r.x, r.y), 0.5, facecolor=cmap(norm(r.generation + 1)))
        ax.add_patch(circle)


def _draw_maze_(maze_env, ax):
    """
    The function to draw maze environment
    Arguments:
        maze_env:   The maze environment configuration.
        ax:         The figure axis instance
    """
    # draw maze walls
    for wall in maze_env.walls:
        line = plt.Line2D((wall.a.x, wall.b.x), (wall.a.y, wall.b.y), color="black", lw=1.5)
        ax.add_line(line)

    # draw start point
    start_circle = plt.Circle((maze_env.agent.location.x, maze_env.agent.location.y),
                              radius=2.5, facecolor="green", edgecolor='w')
    ax.add_patch(start_circle)

    # draw exit point
    exit_circle = plt.Circle((maze_env.exit_point.x, maze_env.exit_point.y),
                             radius=2.5, facecolor=(1.0, 0.2, 0.0), edgecolor='w')

    ax.add_patch(exit_circle)
    #ax.scatter([maze_env.exit_point.x], [maze_env.exit_point.y], s=320, marker="*",
    #           facecolor=(1.0, 0.2, 0.0), edgecolor='w', zorder=3)


if __name__ == "__main__":

    # read command line parameters
    parser = argparse.ArgumentParser(description="The maze experiment visualizer.")
    parser.add_argument('-m', '--maze', choices=["medium", "hard"], help='The maze configuration to use.')
    parser.add_argument('-r', '--records', required=False, default=None, help='The records file.')
    parser.add_argument('-o', '--output', required=False, default="output", help='The file to store the plot.')
    parser.add_argument('-e', '--empty', action="store_true", help='Flag to indicate no records')
    parser.add_argument('-a', '--archive', action="store_true", help='Flag to indicate archive records')

    args = parser.parse_args()
    print(args.maze)
    print(args.records)
    print(args.output)
    print(args.empty)

    # read maze environment
    maze_config = f"/Users/didrik/Documents/Master/Neuroevolution/src/simulation/environments/maze/{args.maze}_maze.txt"
    maze_env = read_environment(maze_config)

    plot_params = {"medium": {"width": 300, "height": 140, "fig_height": 7},
                   "hard": {"width": 205, "height": 205, "fig_height": 7}}
    params = plot_params[args.maze]

    # read agents records
    records = None
    if not args.empty:
        if args.records is not None:
            rs = AgentRecordStore()
            rs.load(args.records)
            records = rs.records

    # render visualization
    draw_maze_records(maze_env,
                      records,
                      filename=args.output,
                      archive=args.archive,
                      width=params["width"],
                      height=params["height"],
                      fig_height=params["fig_height"])
