import matplotlib.pyplot as plt
import argparse

from simulation.environments.maze.agent import AgentRecordStore
from simulation.environments.maze.maze_environment import read_environment


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
        norm = plt.Normalize(1, 1500)

        if not archive:
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                              ax=ax, fraction=0.02, pad=-0.00001)

            cb.set_label(label="Generation", size=20, weight="bold")
            cb.ax.tick_params(labelsize=15)

        # draw species
        _draw_records(records=records, ax=ax, cmap=cmap, norm=norm, archive=archive)

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


def _draw_records(records, ax, cmap, norm, archive):
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
                              radius=3.0, facecolor="black", edgecolor='w')
    ax.add_patch(start_circle)

    # draw exit point
    ax.scatter([maze_env.exit_point.x], [maze_env.exit_point.y], s=300, marker="*",
               facecolor="#e81313")


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

    import glob
    import os

    treatments = ["performance",
                  "performance-beh_div", "performance-hamming",
                  "performance-modularity", "performance-mod_div",
                  "performance-rep_div_cka", "performance-rep_div_cca"]

    local_dir = os.path.dirname(os.path.realpath(__file__))

    treatment_data = {}
    for treatment in treatments:
        print(treatment)

        output_file = os.path.join(local_dir, f"results/plots/mazerobot-{args.maze}/{treatment}_records_{args.maze}")
        results_data_dir = os.path.join(local_dir, f"results/data/mazerobot-{args.maze}/{treatment}/*/")
        results_data_run_dirs = glob.glob(results_data_dir)

        rs = AgentRecordStore()

        for data_path in results_data_run_dirs:

            file = f"agent_record_data.pickle"
            data_file = os.path.join(data_path, file)
            print(f"    - {data_file[-28:-25]}")
            curr_rs = AgentRecordStore()
            curr_rs.load(data_file)
            rs.records.extend(curr_rs.records)

        # render visualization
        draw_maze_records(maze_env,
                          rs.records,
                          filename=output_file,
                          archive=args.archive,
                          width=params["width"],
                          height=params["height"],
                          fig_height=params["fig_height"])
