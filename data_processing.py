from helpers.decorators import array_equals
from math import sqrt


@array_equals
def dot_product(list_1: list, list_2: list) -> float:
    sum_result = 0
    for i in range(len(list_1)):
        sum_result += list_1[i] * list_2[i]
    return sum_result


def vector_length(vector: list) -> float:

    def squared(n):
        return n * n

    squared_vals = map(squared, vector)
    return sum(squared_vals)


@array_equals
def cosine_sim(list_1: list, list_2: list):
    dot_prod_val = dot_product(list_1, list_2)
    vec_1_len = vector_length(list_1)
    vec_2_len = vector_length(list_2)

    return dot_prod_val / (vec_1_len + vec_2_len)


def euclidean(point_x: list, point_y: list):
    return sqrt((point_x[0] - point_y[0]) ** 2 + (point_x[1] - point_y[1]))


def mean(nums: list):
    return sum(nums) / len(nums)


def standard_deviation(nums: list):
    m = mean(nums)
    sub = [(x - m) ** 2 for x in nums]
    return sqrt(sub / len(nums) - 1)


def covarience(x: list, y: list):
    mean_x = mean(x)
    mean_y = mean(y)

    l1 = [(num - mean_x) ** 2 for num in x]
    l2 = [(num - mean_y) ** 2 for num in y]

    dn = dot_product(l1, l2)
    return dn / (len(l1) - 1)


def correlation(x: list, y: list):
    std_x = standard_deviation(x)
    std_y = standard_deviation(y)
    cov = covarience(x, y)
    return cov / (std_x * std_y)


@array_equals
def pair_binary_vector(vector_1: list, vector_2: list) -> dict:
    data = {"11": 0, "10": 0, "00": 0, "01": 0}
    for i in range(len(vector_1)):
        data[f'{vector_1[i]}{vector_2[i]}'] += 1

    return data


def binary_sim(vector_1: list, vector_2: list):
    paired: dict = pair_binary_vector(vector_1, vector_2)
    result = (paired["11"] + paired["00"]) / sum(list(paired.values()))
    return result


def jaccard_coefficient(vector_1: list, vector_2: list):
    paired: dict = pair_binary_vector(vector_1, vector_2)
    result = paired["11"] / (paired["01"] + paired["10"] + paired["11"])
    return result
