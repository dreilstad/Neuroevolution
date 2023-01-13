import os
import shutil
import argparse
from pathlib import Path

def move_data():
    local_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = "/Users/didrik/Documents/Master/Neuroevolution/src/results/data/mazerobot/performance-linear_cka"
    move_to_dir = "/Users/didrik/Documents/Master/Neuroevolution/src/results/data/mazerobot/performance-linear_cka_old"

    files_to_move = []
    for exp in os.listdir(data_dir):
        # get path to experiments 000, 001, ... for treatment
        if "." in exp or int(exp) > 49:
            continue

        file_to_move = os.path.join(data_dir, f"{exp}/agent_record_data.pickle")
        move_to_file = os.path.join(move_to_dir, f"{exp}/agent_record_data.pickle")
        os.rename(file_to_move, move_to_file)


if __name__=="__main__":
    move_data()