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


def arithmetic_crossover_real(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    alpha = 0.5
    for k in range(offspring_size[0]):
        parent1 = parents[k % parents.shape[0]]
        parent2 = parents[(k + 1) % parents.shape[0]]
        child = (1 - alpha) * parent1 + alpha * parent2
        offspring[k, :] = child
    return offspring


def linear_crossover_real(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        c1 = 0.5 * p1 + 0.5 * p2
        c2 = 1.5 * p1 - 0.5 * p2
        c3 = -0.5 * p1 + 1.5 * p2
        children = [c1, c2, c3]
        selected = np.random.choice(3, size=1)[0]
        offspring[k, :] = children[selected]
    return offspring


def blend_crossover_alpha_real(parents, offspring_size, ga_instance, alpha=0.5):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        child = []
        for v1, v2 in zip(p1, p2):
            d = abs(v1 - v2)
            low = min(v1, v2) - alpha * d
            high = max(v1, v2) + alpha * d
            val = np.random.uniform(low, high)
            child.append(np.clip(val, ga_instance.random_mutation_min_val, ga_instance.random_mutation_max_val))
        offspring[k, :] = child
    return offspring


def blend_crossover_alpha_beta_real(parents, offspring_size, ga_instance, alpha=0.5, beta=0.5):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        child = []
        for v1, v2 in zip(p1, p2):
            d = abs(v1 - v2)
            low = min(v1, v2) - alpha * d
            high = max(v1, v2) + beta * d
            val = np.random.uniform(low, high)
            child.append(np.clip(val, ga_instance.random_mutation_min_val, ga_instance.random_mutation_max_val))
        offspring[k, :] = child
    return offspring


def average_crossover_real(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        child = (p1 + p2) / 2
        offspring[k, :] = child
    return offspring
