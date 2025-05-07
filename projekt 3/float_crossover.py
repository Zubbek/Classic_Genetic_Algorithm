import random
import numpy as np

def arithmetic_crossover_real(parents, offspring_size, ga_instance, alpha=0.5):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        parent1 = parents[k % parents.shape[0]]
        parent2 = parents[(k + 1) % parents.shape[0]]
        offspring[k, :] = (1 - alpha) * parent1 + alpha * parent2
    return offspring

def linear_crossover_real(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    idx = 0
    for k in range(0, offspring_size[0], 2):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        c1 = 0.5 * p1 + 0.5 * p2
        c2 = 1.5 * p1 - 0.5 * p2
        c3 = -0.5 * p1 + 1.5 * p2
        chosen = random.sample([c1, c2, c3], 2)
        offspring[idx] = np.clip(chosen[0], ga_instance.init_range_low, ga_instance.init_range_high)
        if idx + 1 < offspring_size[0]:
            offspring[idx + 1] = np.clip(chosen[1], ga_instance.init_range_low, ga_instance.init_range_high)
        idx += 2
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
            child.append(random.uniform(low, high))
        offspring[k, :] = np.clip(child, ga_instance.init_range_low, ga_instance.init_range_high)
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
            child.append(random.uniform(low, high))
        offspring[k, :] = np.clip(child, ga_instance.init_range_low, ga_instance.init_range_high)
    return offspring


def average_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        offspring[k, :] = (p1 + p2) / 2
    return offspring



def make_blend_crossover_alpha_beta(alpha=0.5, beta=0.5):
    def crossover_func(parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0]):
            p1 = parents[k % parents.shape[0]]
            p2 = parents[(k + 1) % parents.shape[0]]
            child = []
            for v1, v2 in zip(p1, p2):
                d = abs(v1 - v2)
                low = min(v1, v2) - alpha * d
                high = max(v1, v2) + beta * d
                child.append(random.uniform(low, high))
            offspring[k, :] = np.clip(child, ga_instance.init_range_low, ga_instance.init_range_high)
        return offspring
    return crossover_func

    
def make_arithmetic_crossover_real(alpha=0.5):
    def crossover_func(parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0]):
            parent1 = parents[k % parents.shape[0]]
            parent2 = parents[(k + 1) % parents.shape[0]]
            offspring[k, :] = (1 - alpha) * parent1 + alpha * parent2
        return offspring
    return crossover_func

    
def make_blend_crossover_alpha_real(alpha=0.5):
    def crossover_func(parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0]):
            p1 = parents[k % parents.shape[0]]
            p2 = parents[(k + 1) % parents.shape[0]]
            child = []
            for v1, v2 in zip(p1, p2):
                d = abs(v1 - v2)
                low = min(v1, v2) - alpha * d
                high = max(v1, v2) + alpha * d
                child.append(random.uniform(low, high))
            offspring[k, :] = np.clip(child, ga_instance.init_range_low, ga_instance.init_range_high)
        return offspring
    return crossover_func