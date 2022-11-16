import warnings
import os
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
#from celluloid import Camera

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


def plot_pareto_2d(checkpoints, filename, domain, label0, label1, max0, max1, min=0, max=-1, invert=True, show=False):
    fitnesses = [[f.fitness for _, f in c.population.items()] for c in checkpoints]
    bests = [c.best_genome.fitness for c in checkpoints]


    fig, ax = plt.subplots(figsize = (10,5))
    ax.set_title(domain + " - Solution Space")

    #camera = Camera(fig)

    if (invert):
        ax.set_xlabel(label1)
        ax.set_ylabel(label0)
    else:
        ax.set_xlabel(label0)
        ax.set_ylabel(label1)

    non_dom_solutions_x = []
    non_dom_solutions_y = []

    # scatter
    for gen in fitnesses:
        dom_solutions_x = []
        dom_solutions_y = []

        for f in gen:
            if f.rank == 0:
                non_dom_solutions_x.append(f.values[0] if (f.values[0] < max0) else max0)
                non_dom_solutions_y.append(f.values[1] if (f.values[1] < max1) else max1)
            else:
                dom_solutions_x.append(f.values[0] if (f.values[0] < max0) else max0)
                dom_solutions_y.append(f.values[1] if (f.values[1] < max1) else max1)

        r = lambda: np.random.randint(0, 255)
        dom_color = "#%02X%02X%02X" % (r(), r(), r())
        non_dom_color = "#%02X%02X%02X" % (r(), r(), r())
        if invert:
            ax.scatter(dom_solutions_y, dom_solutions_x, s=3, c=dom_color)
            ax.scatter(non_dom_solutions_y, non_dom_solutions_x, s=3, c=non_dom_color)
        else:
            ax.scatter(dom_solutions_x, dom_solutions_y, s=3, c=dom_color)
            ax.scatter(non_dom_solutions_x, non_dom_solutions_y, s=3, c=non_dom_color)

        """
        # triangulation
        try:
            tri = Delaunay(list(zip(non_dom_solutions_y,non_dom_solutions_x)))
        except QhullError:
            break

        for t in tri.simplices:
            x = [gen[i].values[0] if (gen[i].values[0] < max0) else max0 for i in t]
            y = [gen[i].values[1] if (gen[i].values[1] < max1) else max1 for i in t]
            ax.fill(y, x, linewidth=0.2, c=non_dom_color, alpha=0.05)
        """

        #camera.snap()

    #x = [f.values[0] if (f.values[0] < max0) else max0 for f in bests]
    #y = [f.values[1] if (f.values[1] < max1) else max1 for f in bests]
    sorted_x_y = [(x,y) for x, y in sorted(zip(non_dom_solutions_x, non_dom_solutions_y), key=lambda pair: pair[0])]
    x_sorted = [x for x, _ in sorted_x_y]
    y_sorted = [y for _, y in sorted_x_y]
    ax.plot(y_sorted, x_sorted, linewidth=1, c="#000000")

    #ax.legend()

    #camera.snap()
    #animation = camera.animate(blit=False, interval=5)
    #animation.save("pareto_front.gif")


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

    dot = graphviz.Digraph('svg', node_attr=node_attrs)

    inputs = set()
    for k in net.input_nodes:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in net.output_nodes:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled',
                      'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    for node, _, _, _, _, links in net.node_evals:
        for i, w in links:
            node_input, output = node, i
            a = node_names.get(output, str(output))
            b = node_names.get(node_input, str(node_input))
            style = 'solid'
            color = 'black'
            width = str(1.0)
            dot.edge(a, b, _attributes={
                     'style': style, 'color': color, 'penwidth': width, 'dir':"none"})

    dot.render(filename)

    return dot

if __name__=="__main__":
    from util import load_checkpoints
    check = load_checkpoints("/Users/didrik/Documents/Master/Neuroevolution/src/checkpoints/retina/performance-hamming/000")
    save_file = "/Users/didrik/Documents/Master/Neuroevolution/src/results/plots/retina/performance-hamming/000/pareto_front_test.png"
    plot_pareto_2d(check, save_file, "Retina",
                             "Task Performance", "Hamming Distance",
                             500.0, 10000.0)