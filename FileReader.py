import numpy as np


def import_data(path):
    """
    Read a text file form the given path and return a np array containing the textfiles contents
    :param path: Path to textfile
    :return: np array with data from textfile
    """
    return np.loadtxt(
        path,
        dtype="float",
        delimiter=','
    )
