import numpy as np


def my_single_point_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        point = np.random.randint(1, parents.shape[1])  # punkt krzyżowania
        offspring[k, :point] = p1[:point]
        offspring[k, point:] = p2[point:]
    return offspring


def my_two_point_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        pt1, pt2 = sorted(np.random.choice(range(1, parents.shape[1]), 2, replace=False))
        offspring[k, :pt1] = p1[:pt1]
        offspring[k, pt1:pt2] = p2[pt1:pt2]
        offspring[k, pt2:] = p1[pt2:]
    return offspring


def my_uniform_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        mask = np.random.rand(parents.shape[1]) < 0.5
        offspring[k] = np.where(mask, p1, p2)
    return offspring


def my_discrete_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        offspring[k] = [np.random.choice([g1, g2]) for g1, g2 in zip(p1, p2)]
    return offspring
