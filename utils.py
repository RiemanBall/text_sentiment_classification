import numpy as np
import pandas as pd

def load_training_data(path='training_label.txt'):
    """
    Reading training set.
    If the file is 'training_label.txt', then we need to read the label information. Otherwise, we don't.
    Data format:
        With label:
            label, separator, text
        Without label:
            text

    Inputs:
    - path: str. The path of the dataset

    Outputs:
    - x: List of str.
    - y: List of 
    """
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        y = [int(line[0]) for line in lines]
        # Start from 2 to skip the separator
        x = [line[2:] for line in lines]

        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]

        return x


def load_testing_data(path='testing_data.txt'):
    """
    Reading testing set.
    
    Data format:
        title
        id1,text1
        id2,text2
            .
            .

    Inputs:
    - path: str. The path of the dataset

    Outputs:
    - X: List of str.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]

    return X


def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0 
    correct = np.sum(outputs == labels)
    return correct