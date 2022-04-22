from pandas import DataFrame
from math import log2
from helpers.merge_sort import merge_sort


def get_frequency(data: DataFrame, class_col: str, val_col: str, value, default=True) -> dict:
    freq = {'counts': {}}
    class_values = data[class_col].unique()
    total = data.shape[0]
    cond = True

    if not default:
        cond = (data[val_col] == value)
        total = 0

    for val in class_values:
        count = data[(data[class_col] == val) & cond].shape[0]
        freq['counts'][val] = count
        total += count

    freq['total'] = total

    return freq


def gini_index(data: DataFrame, class_col, val_col, value, default=True):
    freqs_data = get_frequency(data, class_col, val_col, value, default)

    total = freqs_data['total']
    freqs = freqs_data['counts']
    result = 1

    for freq in freqs:
        result -= (freq / total) ** 2

    return - result


def entropy(data: DataFrame, class_col, val_col, value, default=True):
    freqs_data = get_frequency(data, class_col, val_col, value, default)

    total = freqs_data['total']
    freqs = freqs_data['counts']
    result = 0

    for freq in freqs:
        c_freq = freq/total
        result -= c_freq * log2(c_freq)

    return result


def gain(data: DataFrame, class_col, val_col, default=True):
    vals = data[val_col].unique()

    entropies = []

    for val in vals:
        dt = data[data[val_col] == val]
        e = entropy(dt, class_col, val_col, val, default)
        entropies.append(e)

    merge_sort(entropies)
    gain = entropies[-1]

    for i in range(len(entropies)-2, -1, -1):
        gain -= entropies[i]

    return gain
