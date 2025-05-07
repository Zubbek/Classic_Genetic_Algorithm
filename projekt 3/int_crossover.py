
import numpy
def single_point_crossover_pygad(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        point = numpy.random.randint(1, parents.shape[1])  # Punkt krzyżowania
        child = numpy.concatenate((parent1[:point], parent2[point:]))
        offspring.append(child)
        idx += 1
    return numpy.array(offspring)

def two_point_crossover_pygad(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        point1, point2 = sorted(numpy.random.choice(range(1, parents.shape[1]), 2, replace=False))
        child = numpy.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        offspring.append(child)
        idx += 1
    return numpy.array(offspring)

def uniform_crossover_pygad(parents, offspring_size, ga_instance, cross_probability=0.7):
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        child = []
        for gene1, gene2 in zip(parent1, parent2):
            if numpy.random.rand() < cross_probability:
                child.append(gene2)
            else:
                child.append(gene1)
        offspring.append(numpy.array(child))
        idx += 1
    return numpy.array(offspring)


def make_uniform_crossover_pygad(cross_probability=0.5):
    def crossover_func(parents, offspring_size, ga_instance):
        offspring = []
        idx = 0
        while len(offspring) < offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            child = []
            for gene1, gene2 in zip(parent1, parent2):
                if numpy.random.rand() < cross_probability:
                    child.append(gene2)
                else:
                    child.append(gene1)
            offspring.append(numpy.array(child))
            idx += 1
        return numpy.array(offspring)
    return crossover_func