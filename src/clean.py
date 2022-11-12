import argparse
import os, shutil


def clean_all_experiments():
    experiments = ["xor", "retina", "mazerobot", "bipedal"]

    for domain in experiments:
        clean_experiment(domain)


def clean_experiment(domain):
    local_dir = os.path.dirname(os.path.realpath(__file__))

    # result dirs
    results_data_dir = os.path.join(local_dir, f"results/data/{domain}")
    _clear_dir(results_data_dir)

    results_plot_dir = os.path.join(local_dir, f"results/plots/{domain}")
    _clear_dir(results_plot_dir)

    # checkpoint dirs
    checkpoint_dir = os.path.join(local_dir, f"checkpoints/{domain}")
    _clear_dir(checkpoint_dir)


def _clear_dir(path):

    try:
        for name in os.listdir(path):
            file_dir_path = os.path.join(path, name)
            try:
                if os.path.isfile(file_dir_path) or os.path.islink(file_dir_path):
                    os.unlink(file_dir_path)
                elif os.path.isdir(file_dir_path):
                    shutil.rmtree(file_dir_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_dir_path, e))
    except FileNotFoundError:
        pass


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Clean experiment run folders",
                                     add_help=True)
    parser.add_argument("-e", "--experiment",
                        type=str,
                        required=False,
                        default="all",
                        choices=["xor", "retina", "mazerobot", "bipedal"],
                        help="Name of experiment to clean, default is all")

    args = parser.parse_args()

    if args.experiment == "all":
        clean_all_experiments()
    else:
        clean_experiment(args.experiment)
