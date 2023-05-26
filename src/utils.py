from os import path
from pathlib import Path

import torch

PROJECT_ROOT = str(Path.cwd().parent)  # project path


def write_data_to_txt(file_path, data):
    if path.exists(file_path):
        with open(file_path, 'a', newline='') as file:
            file.write(data)
    else:  # Create the file
        with open(file_path, 'w') as file:
            file.write(data)


def set_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
