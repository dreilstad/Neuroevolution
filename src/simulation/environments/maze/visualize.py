import random
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import simulation.environments.maze.geometry as geometry
from simulation.environments.maze.agent import AgentRecordStore
from simulation.environments.maze.maze_environment import read_environment


def draw_agent_path(maze_env, path_points, genome, filename=None, view=False, show_axes=False, width=400, height=400,
                    fig_height=4):
    """
    The function to draw path of the maze solver agent through the maze.
    Arguments:
        maze_env:       The maze environment configuration.
        path_points:    The list of agent positions during simulation.
        genome:         The genome of solver agent.
        filename:       The name of file to store plot.
        view:           The flag to indicate whether to view plot.
        width:          The width of drawing in pixels
        height:         The height of drawing in pixels
        fig_height:      The plot figure height in inches
    """
    # initialize plotting
    fig, ax = plt.subplots()
    fig.set_dpi(100)
    fig_width = fig_height * (float(width) / float(height)) - 0.2
    # print("Plot figure width: %.1f, height: %.1f" % (fig_width, fig_height))
    fig.set_size_inches(fig_width, fig_height)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    ax.set_title('Genome ID: %s, Path Length: %d' % (genome.GetID(), len(path_points)))
    # draw path
    for p in path_points:
        circle = plt.Circle((p.x, p.y), 2.0, facecolor='b')
        ax.add_patch(circle)

    # draw maze
    _draw_maze_(maze_env, ax)

    # turn off axis rendering
    if not show_axes:
        ax.axis('off')

    # Invert Y axis to have coordinates origin at the top left
    ax.invert_yaxis()

    # Save figure to file
    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_maze_records(maze_env, records, best_threshold=0.8, filename=None, show_axes=False, width=140,
                      height=300, fig_height=7):
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
    # find the distance threshold for the best species
    dist_threshold = maze_env.agent_distance_to_exit() * (1.0 - best_threshold)

    # initialize plotting
    fig = plt.figure()
    ax = plt.gca()
    #fig.set_dpi(300)
    # print("Plot figure width: %.1f, height: %.1f" % (fig_width, fig_height))
    fig_width = fig_height * (float(width) / float(height))
    fig.set_size_inches(fig_width, fig_height)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("equal")

    cmap = plt.get_cmap("gnuplot2")
    norm = plt.Normalize(records[0].generation + 1, records[-1].generation + 1)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, fraction=0.02, pad=-0.00001)

    cb.set_label(label="Generation", size=20, weight="bold")
    cb.ax.tick_params(labelsize=15)

    # draw species
    _draw_species_(records=records, ax=ax, cmap=cmap, norm=norm)

    # draw maze
    _draw_maze_(maze_env, ax)

    # turn off axis rendering
    if not show_axes:
        ax.axis('off')
    # Invert Y axis to have coordinates origin at the top left
    ax.invert_yaxis()

    # Save figure to file
    if filename is not None:
        plt.savefig(filename + ".pdf", bbox_inches="tight")

    plt.close()


def _draw_species_(records, ax, cmap, norm):
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


if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The maze experiment visualizer.")
    parser.add_argument('-m', '--maze', default='medium', help='The maze configuration to use.')
    parser.add_argument('-r', '--records', help='The records file.')
    parser.add_argument('-o', '--output', help='The file to store the plot.')
    parser.add_argument('--width', type=int, default=200, help='The width of the subplot')
    parser.add_argument('--height', type=int, default=200, help='The height of the subplot')
    parser.add_argument('--fig_height', type=float, default=7, help='The height of the plot figure')
    parser.add_argument('--show_axes', type=bool, default=False, help='The flag to indicate whether to show plot axes.')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    if not (args.maze == 'medium' or args.maze == 'hard'):
        print('Unsupported maze configuration: %s' % args.maze)
        exit(1)

    # read maze environment
    maze_env_config = os.path.join(local_dir, '%s_maze.txt' % args.maze)
    maze_env = read_environment(maze_env_config)

    # read agents records
    rs = AgentRecordStore()
    rs.load(args.records)

    # render visualization
    random.seed(42)
    draw_maze_records(maze_env,
                      rs.records,
                      width=args.width,
                      height=args.height,
                      fig_height=args.fig_height,
                      view=True,
                      show_axes=args.show_axes,
                      filename=args.output)