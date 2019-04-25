from os.path import expanduser, join
import numpy as np


def pform(path, *names, sep= ''):
    """formats a path as `path` followed by `names` joined with `sep`."""
    return join(expanduser(path), sep.join(map(str, names)))


def load_txt(filename):
    """yields lines from text file."""
    with open(filename) as file:
        yield from (line[:-1] for line in file)


def save_txt(filename, lines, split=""):
    """writes lines to text file."""
    with open(filename, 'w') as file:
        for line in lines:
            print(line+split, file= file)

