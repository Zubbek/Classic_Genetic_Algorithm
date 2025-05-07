import numpy as np
def gaussian_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        gene_idx = np.random.randint(offspring.shape[1])
        offspring[idx, gene_idx] += np.random.normal(0, 0.1)  # dobierz sigma
    return offspring

def swap_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        gene_indices = np.random.choice(range(offspring.shape[1]), 2, replace=False)
        offspring[idx, gene_indices[0]], offspring[idx, gene_indices[1]] = \
            offspring[idx, gene_indices[1]], offspring[idx, gene_indices[0]]
    return offspring