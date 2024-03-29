import os
import glob
import argparse
import neat

from simulation.retina_simulation import RetinaSimulator, HardRetinaSimulator
from simulation.maze_simulation import MediumMazeSimulator, HardMazeSimulator
from simulation.bipedal_walker_simulation import BipedalWalkerSimulator
from simulation.tartarus_simulation import TartarusSimulator, DeceptiveTartarusSimulator


def validate_input(args):
    config = validate_config_file(args.config_file)
    objectives = validate_objectives(args.objectives)
    domain = validate_domain(args.domain)

    return domain, config, objectives, args.num_generations


def validate_config_file(filename):
    local_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(local_dir, "config/" + filename)
    if not os.path.exists(config_path):
        raise argparse.ArgumentTypeError(f"'{filename}' does not exist")

    return config_path


def validate_domain(domain):
    simulators = {"retina": RetinaSimulator,
                  "retina-hard": HardRetinaSimulator,
                  "bipedal": BipedalWalkerSimulator,
                  "mazerobot-medium": MediumMazeSimulator,
                  "mazerobot-hard": HardMazeSimulator,
                  "tartarus": TartarusSimulator,
                  "tartarus-deceptive": DeceptiveTartarusSimulator}

    if domain not in simulators:
        raise RuntimeError("Domain is no valid")

    return simulators[domain]


def validate_objectives(objectives):
    valid_objectives = ["performance",
                        "hamming", "beh_div",
                        "mod_div", "modularity",
                        "rep_div_cka", "rep_div_cca"]
    for objective in objectives:
        if objective not in valid_objectives:
            raise RuntimeError(f"Objective '{objective}' is not valid")

    return objectives


def clear_checkpoints(path, save_last=False):
    checkpoint_path = os.path.join(path, "*.pickle")

    files = glob.glob(checkpoint_path)
    files.sort(key=os.path.getmtime)

    if save_last:
        files = files[:-1]

    for f in files:
        os.remove(f)


def load_checkpoints(folder):
    print("Loading checkpoints from {0}...".format(folder))
    # load generations from file
    checkpoints = []
    local_dir = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(local_dir, folder)
    files = os.listdir(folder)
    # progress bar vars
    step = len(files)/46
    t = 0
    print('[', end='', flush=True)
    for filename in files:
        # load checkpoint and append to list
        checkpoint = neat.Checkpointer.restore_checkpoint(os.path.join(folder, filename))
        checkpoints.append(checkpoint)
        # update progress bar
        t += 1
        if (t > step):
            t -= step
            print('.', end='', flush=True)
    print(']')
    # Sort checkpoints by generation id
    checkpoints.sort(key = lambda g: g.generation)
    return checkpoints


def load_last_checkpoint(folder):
    checkpoint_path = os.path.join(folder, "*.pickle")
    checkpoints = glob.glob(checkpoint_path)

    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


def make_new_run_folders(domain, objectives):
    local_dir = os.path.dirname(os.path.realpath(__file__))

    objectives_string = "-".join(objectives)

    # result dirs
    results_data_dir = os.path.join(local_dir, f"results/data/{domain}/{objectives_string}")
    results_data_dir_path = make_new_run_folder(results_data_dir)

    results_plot_dir = os.path.join(local_dir, f"results/plots/{domain}/{objectives_string}")
    results_plot_dir_path = make_new_run_folder(results_plot_dir)

    # checkpoint dirs
    checkpoint_dir = os.path.join(local_dir, f"checkpoints/{domain}/{objectives_string}")
    checkpoint_dir_path = make_new_run_folder(checkpoint_dir)

    return results_data_dir_path, results_plot_dir_path, checkpoint_dir_path


def make_new_run_folder(dir_path):

    # find folders in directory
    folders = [x[0] for x in os.walk(dir_path)][1:]

    # if directory is empty create new run folder with index 0,
    # else create new folder with incremented index from latest run
    if len(folders) == 0:
        new_run_dir = os.path.join(dir_path, str(0).zfill(3))
        os.makedirs(new_run_dir)
    else:
        latest_folder = sorted(folders)[-1]
        latest_experiment_nr = os.path.basename(os.path.normpath(latest_folder))
        new_experiment_nr = int(latest_experiment_nr) + 1
        new_run_dir = os.path.join(dir_path, str(new_experiment_nr).zfill(3))
        os.makedirs(new_run_dir)

    return new_run_dir


def get_empty_run_folders(domain, objectives):
    local_dir = os.path.dirname(os.path.realpath(__file__))
    objectives_string = "-".join(objectives)

    results_data_dir = os.path.join(local_dir, f"results/data/{domain}/{objectives_string}/*/")
    results_plots_dir = os.path.join(local_dir, f"results/plots/{domain}/{objectives_string}/*/")
    checkpoint_dir = os.path.join(local_dir, f"checkpoints/{domain}/{objectives_string}/*/")

    results_data_run_dirs = glob.glob(results_data_dir)
    results_plots_run_dirs = glob.glob(results_plots_dir)
    checkpoint_run_dirs = glob.glob(checkpoint_dir)

    unfinished_runs = []
    for i in range(len(checkpoint_run_dirs)):
        if len(os.listdir(results_data_run_dirs[i])) == 0:
            unfinished_runs.append((results_data_run_dirs[i],
                                    results_plots_run_dirs[i],
                                    checkpoint_run_dirs[i]))

    return unfinished_runs

