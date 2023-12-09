import os
import neat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.path import Path
import matplotlib.patches as patches
import glob
import warnings
import graphviz
import imageio

from simulation.bipedal_walker_simulation import BipedalWalkerSimulator

preamble = [r'\usepackage[T1]{fontenc}',
            r'\usepackage{amsmath}',
            r'\usepackage{txfonts}',
            r'\usepackage{textcomp}']
matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
matplotlib.rc('text.latex', preamble="\n".join(preamble))

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


def _get_aggregated_runtime_data(domain):

    treatments = ["performance",
                  "performance-beh_div", "performance-hamming",
                  "performance-modularity", "performance-mod_div",
                  "performance-rep_div_cka", "performance-rep_div_cca"]

    local_dir = os.path.dirname(os.path.realpath(__file__))

    treatment_data = {}
    for treatment in treatments:
        results_data_dir = os.path.join(local_dir, f"results/data/{domain}/{treatment}/*/")
        results_data_run_dirs = glob.glob(results_data_dir)

        scores = []
        for data_path in results_data_run_dirs:

            file = f"avg_gen_time_{data_path[-4:-1]}.txt"
            data_file = os.path.join(data_path, file)

            with open(data_file, "r") as f:
                last_line = f.readlines()[-1]

            value = float(last_line)
            scores.append(value)

        print(treatment)
        print(scores)
        print()
        mean = np.mean(scores)
        error = np.std(scores)

        treatment_data[treatment] = [mean, error]

    print(treatment_data)
    return treatment_data


def _get_aggregated_performance_data():

    domains = ["retina", "retina-hard",
               "tartarus", "tartarus-deceptive",
               "mazerobot-medium", "mazerobot-hard",
               "bipedal"]

    treatments = ["performance",
                  "performance-beh_div", "performance-hamming",
                  "performance-modularity", "performance-mod_div",
                  "performance-rep_div_cka", "performance-rep_div_cca"]

    local_dir = os.path.dirname(os.path.realpath(__file__))

    data = []
    for domain in domains:
        treatment_data = []
        print(domain)
        for treatment in treatments:
            results_data_dir = os.path.join(local_dir, f"results/data/{domain}/{treatment}/*/")
            results_data_run_dirs = glob.glob(results_data_dir)

            scores = []
            for data_path in results_data_run_dirs:

                file = f"best_fitness_{data_path[-4:-1]}.dat"
                data_file = os.path.join(data_path, file)

                with open(data_file, "r") as f:
                    last_line = f.readlines()[-1]

                value = float(last_line.split()[-1])
                scores.append(value)

            median = np.median(scores)
            if np.isnan(median):
                median = 0.0

            if domain == "tartarus" or domain == "tartarus-deceptive":
                print(treatment)
                print(f"- {max(scores)}")
                print(f"- {results_data_run_dirs[np.argmax(scores)][-4:-1]}")

            treatment_data.append(median)

        data.append(treatment_data)

    return data


def _get_aggregated_success_rate_data():

    domains = ["retina", "retina-hard",
               "tartarus", "tartarus-deceptive",
               "mazerobot-medium", "mazerobot-hard",
               "bipedal"]

    treatments = ["performance",
                  "performance-beh_div", "performance-hamming",
                  "performance-modularity", "performance-mod_div",
                  "performance-rep_div_cka", "performance-rep_div_cca"]

    success_criterion = {"retina": 1.0,
                         "retina-hard": 1.0,
                         "tartarus": 10.0,
                         "tartarus-deceptive": 8.0,
                         "mazerobot-medium": 1.0,
                         "mazerobot-hard": 1.0,
                         "bipedal": 300.0}

    local_dir = os.path.dirname(os.path.realpath(__file__))

    data = []
    for domain in domains:
        treatment_data = []
        for treatment in treatments:
            results_data_dir = os.path.join(local_dir, f"results/data/{domain}/{treatment}/*/")
            results_data_run_dirs = glob.glob(results_data_dir)

            max_score_reached = 0.0
            for data_path in results_data_run_dirs:

                file = f"best_fitness_{data_path[-4:-1]}.dat"
                data_file = os.path.join(data_path, file)

                with open(data_file, "r") as f:
                    last_line = f.readlines()[-1]

                value = float(last_line.split()[-1])

                if value >= success_criterion[domain]:
                    max_score_reached += 1.0

            success_rate = 0.0
            if len(results_data_run_dirs) > 0:
                success_rate = max_score_reached / len(results_data_run_dirs)

            treatment_data.append(success_rate)

        data.append(treatment_data)

    for domain, d in zip(domains, data):
        print(domain)
        print(d)
        print()

    return data


def parallel_coordinates_plot(success_rate=False):

    treatments = ["PA", "Novelty", "Hamming", "Mod", "ModDiv", "CKA", "CCA"]
    ynames = ["Retina 2x2", "Retina 3x3", "Tartarus", "Deceptive-Tartarus",
              "Medium-Maze", "Hard-Maze", "Bipedal-Walker"]

    if success_rate:
        ys = _get_aggregated_success_rate_data()
        ymins = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ymaxs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    else:
        ys = _get_aggregated_performance_data()
        ymins = np.array([0.89, 0.89, 4.0, 1.6, 0.85, 0.6, 200.0])
        ymaxs = np.array([1.0, 1.0, 6.0, 2.4, 1.0, 1.0, 260.0])

    # organize the data
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    ys = np.array(ys)
    ys = ys.T
    print(ys)
    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    fig, host = plt.subplots(figsize=(18, 8))

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=15)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=25)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()

    if success_rate:
        host.set_title('Success rate', fontsize=35, pad=12)
    else:
        host.set_title('Median best performance', fontsize=35, pad=12)

    colors = {"PA":"#ffe119",
              "Novelty":"#4daf4a",
              "Hamming":"#377eb8",
              "Mod":"#984ea3",
              "ModDiv":"#ff7f00",
              "CKA":"#e41a1c",
              "CCA":"#000000"}

    linestyle = {"PA":(0, (5,10)),
              "Novelty":"--",
              "Hamming":"-.",
              "Mod":":",
              "ModDiv":"-",
              "CKA":(0, (3,5,1,5)),
              "CCA":(0, (1,5))}

    legend_handles = [None for _ in treatments]
    for j in range(ys.shape[0]):
        print(treatments[j])
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none',
                                  lw=3, edgecolor=colors[treatments[j]])
        legend_handles[j] = patch
        host.add_patch(patch)

    host.legend(legend_handles, treatments, fontsize=22,
                loc='lower center', bbox_to_anchor=(0.5, -0.18),
                ncol=len(treatments), fancybox=True, shadow=True)
    plt.tight_layout()
    if success_rate:
        plt.savefig(f"results/plots/success_rate_all.pdf", dpi=300)
    else:
        plt.savefig(f"results/plots/median_final_performance_all.pdf", dpi=300)
    plt.show()


def barplot_runtime_overhead(treatments, domain):

    # width of the bars

    bar_width = 0.6

    domain_labels = {"retina": "Retina 2x2",
                     "retina-hard": "Retina 3x3",
                     "bipedal": "Bipedal Walker",
                     "tartarus": "Tartarus",
                     "tartarus-deceptive": "Deceptive Tartarus",
                     "mazerobot-medium": "Medium Maze",
                     "mazerobot-hard": "Hard Maze"}

    labels = {"performance":"PA",
              "performance-beh_div":"Novelty",
              "performance-hamming":"Hamming",
              "performance-modularity":"Mod",
              "performance-mod_div":"ModDiv",
              "performance-rep_div_cka":"CKA",
              "performance-rep_div_cca":"CCA"}

    colors = {"PA":"#ffe119",
              "Novelty":"#4daf4a",
              "Hamming":"#377eb8",
              "Mod":"#984ea3",
              "ModDiv":"#ff7f00",
              "CKA":"#e41a1c",
              "CCA":"#000000"}

    x = []
    y = []
    error = []
    color = []
    order = ["PA", "Novelty", "Hamming", "Mod", "ModDiv", "CKA", "CCA"]
    for treatment in labels.keys():
        x.append(labels[treatment])
        y.append(treatments[treatment][0])
        error.append(treatments[treatment][1])
        color.append(colors[labels[treatment]])

    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(18)

    ax.bar(x, y,
           width=bar_width,
           color=color,
           align="center",
           alpha=0.8,
           edgecolor='black',
           yerr=error,
           capsize=5,
           label=order)

    # general layout
    ax.set_ylabel("Time per generation", fontsize=45)
    ax.set_title(f"{domain_labels[domain]} - Computational run-time overhead", fontsize=45)
    ax.tick_params(axis='both', which='both', labelsize=35)

    # Show graphic
    plt.tight_layout()
    plt.savefig(f"results/runtimes/{domain}_runtime.pdf", dpi=300)
    plt.show()


def create_sequence(filename, frames, start_frame=30, nrows=2, ncols=6):
    print(len(frames))

    fig, axes = plt.subplots(figsize=(18, 6), ncols=ncols, nrows=nrows)
    img_frames = frames[::8]
    for ind, frame in enumerate(img_frames[start_frame:start_frame + (nrows * ncols)]):
        axes.ravel()[ind].imshow(frame[100:303, :250])
        axes.ravel()[ind].set_title(f"Frame {ind + 1}", fontsize=30)
        axes.ravel()[ind].set_axis_off()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    with imageio.get_writer(filename[:-3] + "gif", mode='I') as writer:
        for frame in frames[::8]:
            writer.append_data(frame[100:303, :250])

    plt.show()

def visualize_bipedal_walker(treatment, run="000"):
    local_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(local_dir, f"checkpoints/bipedal/{treatment}/{run}/*.pickle")
    checkpoint = glob.glob(checkpoint_dir)[0]

    pop = neat.Checkpointer.restore_checkpoint(checkpoint)
    neural_network = neat.nn.FeedForwardNetwork.create(pop.best_genome, pop.config)

    sim = BipedalWalkerSimulator(["performance"], "bipedal")
    frames = sim.visualize(neural_network)
    create_sequence(f"bipedal_walker_{treatment}_{run}.png", frames)


if __name__ == "__main__":
    #_get_aggregated_performance_data()
    #visualize_bipedal_walker("performance", run="000")
    #visualize_bipedal_walker("performance-hamming", run="015")
    visualize_bipedal_walker("performance-beh_div", run=str(np.random.randint(0, 50)).zfill(3))
    #visualize_bipedal_walker("performance-modularity", run="041")
    #visualize_bipedal_walker("performance-mod_div", run="034")
    #visualize_bipedal_walker("performance-rep_div_cka", run="032")
    #visualize_bipedal_walker("performance-rep_div_cca", run="022")
