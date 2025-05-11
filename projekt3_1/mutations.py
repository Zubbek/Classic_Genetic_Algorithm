import numpy as np

# ====== Mutacje ======


def swap_mutation(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        idx1, idx2 = np.random.choice(offspring.shape[1], 2, replace=False)
        offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]
    return offspring


def boundary_mutation(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        if np.random.rand() < 0.5:
            offspring[i][0] = 1 - int(round(offspring[i][0]))
        else:
            offspring[i][-1] = 1 - int(round(offspring[i][-1]))
    return offspring


def one_point_mutation(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        idx = np.random.randint(offspring.shape[1])
        offspring[i][idx] = 1 - int(round(offspring[i][idx]))
    return offspring


def two_point_mutation(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        idx1, idx2 = np.random.choice(offspring.shape[1], 2, replace=False)
        offspring[i][idx1] = 1 - int(round(offspring[i][idx1]))
        offspring[i][idx2] = 1 - int(round(offspring[i][idx2]))
    return offspring


def gaussian_mutation(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        gene_idx = np.random.randint(offspring.shape[1])
        offspring[i, gene_idx] += np.random.normal(loc=0.0, scale=1.0)
    return offspring
