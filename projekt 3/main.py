#na podstawie przykładu: https://pypi.org/project/pygad/1.0.18/
import logging
import pygad
import numpy
import benchmark_functions as bf
#from opfunu.cec_based.cec2014 import F132014
import numpy as np

from mutayion import gaussian_mutation,swap_mutation
from int_crossover import single_point_crossover_pygad,two_point_crossover_pygad,uniform_crossover_pygad
from float_crossover import arithmetic_crossover_real,linear_crossover_real,blend_crossover_alpha_real,blend_crossover_alpha_beta_real,average_crossover,make_arithmetic_crossover_real,make_blend_crossover_alpha_beta,make_blend_crossover_alpha_real
#Konfiguracja algorytmu genetycznego
def get_function(name, ndim):
    """Zwraca wybraną funkcję testową na podstawie jej nazwy."""
    if name == "Hypersphere":
        return bf.Hypersphere(n_dimensions=ndim)
    elif name == "Shifted and Rotated HappyCat Function":
        return bf.Hypersphere(n_dimensions=ndim)

        # func = F132014(ndim=ndim)
        # return func.evaluate
        
num_vars = 3
bits_per_var = 20
num_genes = num_vars * bits_per_var

func = bf.Ackley(n_dimensions=num_genes)

num_generations = 100
sol_per_pop = 80
num_parents_mating = 50
#boundary = func.suggested_bounds() #możemy wziąć stąd zakresy
init_range_low = 0
init_range_high = 2
gene_type = int


mutation_num_genes = 1
parent_selection_type = "tournament"
# Zmieniaj parametr parent_selection_type na:
# "tournament"
# "rws" (roulette wheel selection)
# "random"

crossover_type = "uniform"
mutation_type = "random"


#Konfiguracja logowania

level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

def on_generation(ga_instance):
    # ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    # ga_instance.logger.info("Best    = {fitness}".format(fitness=1./solution_fitness))
    # ga_instance.logger.info("Individual    = {solution}".format(solution=repr(solution)))

    tmp = [1./x for x in ga_instance.last_generation_fitness] #ponownie odwrotność by zrobić sobie dobre statystyki

    # ga_instance.logger.info("Min    = {min}".format(min=numpy.min(tmp)))
    # ga_instance.logger.info("Max    = {max}".format(max=numpy.max(tmp)))
    # ga_instance.logger.info("Average    = {average}".format(average=numpy.average(tmp)))
    # ga_instance.logger.info("Std    = {std}".format(std=numpy.std(tmp)))
    # ga_instance.logger.info("\r\n")




def decodeInd(individual):
    bits_per_var = 20
    num_vars = 2
    bounds = func.suggested_bounds()
        
    decoded = []
    for i in range(num_vars):
        bits = individual[i * bits_per_var:(i + 1) * bits_per_var]
        bitstring = ''.join(str(int(b)) for b in bits)
        value = int(bitstring, 2)
        min_val = bounds[0][i]
        max_val = bounds[1][i]

        max_binary = 2 ** bits_per_var - 1
        norm_val = min_val + (value / max_binary) * (max_val - min_val)
        decoded.append(norm_val)
    return decoded


def fitnessFunction(ga_instance, solution, solution_idx):
    ind = decodeInd(solution)
    result = (ind[0] + 2*ind[1] - 7)**2 + (2*ind[0] + ind[1] - 5)**2
    return 1.0 / result if result != 0 else float("inf")


def fitness_func(ga_instance, solution, solution_idx):
    fitness = func(solution)
    return 1./fitness

#Właściwy algorytm genetyczny
custom_crossover = make_blend_crossover_alpha_beta(alpha=0.3, beta=0.7)
ga_instance = pygad.GA(
            num_generations=num_generations,
            sol_per_pop=sol_per_pop,
            num_parents_mating=num_parents_mating,
            num_genes=num_genes,
            gene_type=float,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            mutation_num_genes=mutation_num_genes,
            parent_selection_type=parent_selection_type,
            crossover_type=average_crossover,#ga_instance = pygad.GA(..., crossover_type=single_point_crossover_pygad, ...)
            #crossover_func=single_point_crossover_pygad,
            mutation_type=mutation_type,
            keep_elitism=1,
            K_tournament=3,
            logger=logger,
            fitness_func=fitness_func,
            on_generation=on_generation,
            parallel_processing=["thread", 4]
        )


ga_instance.run()


best = ga_instance.best_solution()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
#print("Parameters of the best solution : {solution}".format(solution=solution))
#print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1./solution_fitness))


# sztuczka: odwracamy my narysował nam się oczekiwany wykres dla problemu minimalizacji
ga_instance.best_solutions_fitness = [1. / x for x in ga_instance.best_solutions_fitness]
ga_instance.plot_fitness()
