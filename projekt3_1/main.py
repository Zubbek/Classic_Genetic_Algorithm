# import logging
# import pygad
# import numpy as np
# import benchmark_functions as bf
# from opfunu.cec_based import F132014
# import matplotlib.pyplot as plt
#
# from crossovers import my_single_point_crossover, my_uniform_crossover, my_discrete_crossover, \
#     my_two_point_crossover
#
#
# # ===== Funkcja dekodująca osobnika binarnego =====
# def decodeInd(individual, bits_per_var=20, var_count=5, x_min=-32.768, x_max=32.768):
#     decoded = []
#     for i in range(var_count):
#         start = i * bits_per_var
#         end = start + bits_per_var
#         binary_slice = individual[start:end]
#         binary_str = ''.join(str(min(1, max(0, int(round(bit))))) for bit in binary_slice)
#         decimal_value = int(binary_str, 2)
#         max_decimal = 2**bits_per_var - 1
#         real_value = x_min + (decimal_value / max_decimal) * (x_max - x_min)
#         decoded.append(real_value)
#     return decoded
#
# # ===== Benchmark funkcja =====
# def get_function(name, ndim):
#     if name == "Hypersphere":
#         return bf.Hypersphere(n_dimensions=ndim)
#     elif name == "Shifted and Rotated HappyCat Function":
#         func = F132014(ndim=ndim)
#         return func.evaluate
#
# # ===== Funkcja przystosowania =====
# def fitnessFunction(individual, num_vars, bits_per_var, is_minimum=False):
#     real_values = decodeInd(individual, bits_per_var, num_vars)
#     fitness = func(real_values)
#     return fitness + 1e-6
#
# # ====== Mutacje ======
# def swap_mutation(offspring, ga_instance):
#     for i in range(offspring.shape[0]):
#         idx1, idx2 = np.random.choice(offspring.shape[1], 2, replace=False)
#         offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]
#     return offspring
#
# mutation_methods = {
#     "random": "random",
#     "swap": swap_mutation
# }
# # ===== Parametry ogólne =====
# num_vars = 5
# bits_per_var = 20
# num_genes = num_vars * bits_per_var
# func = get_function("Hypersphere", num_vars)
# num_generations = 50
# is_minimum = True
# use_binary = True
#
# # ===== Parametry GA (stałe) =====
# common_args = dict(
#     sol_per_pop=80,
#     num_parents_mating=50,
#     num_genes=num_genes,
#     mutation_num_genes=1,
#     keep_elitism=1,
#     init_range_low=0,
#     init_range_high=1,
#     random_mutation_max_val=1,
#     random_mutation_min_val=0
# )
#
#
# # ===== Eksperyment główny =====
# selection_methods = ["tournament", "rws", "random"]
# crossover_methods = {
#     # Wbudowane w PyGAD (stringi)
#     "single_point": "single_point",
#     "two_points": "two_points",
#     "uniform": "uniform",
#
#     # Twoje własne funkcje
#     "my_single_point": my_single_point_crossover,
#     "my_two_point": my_two_point_crossover,
#     "my_uniform": my_uniform_crossover,
#     "my_discrete": my_discrete_crossover
# }
#
#
# experiment_results = {}
# is_minimum = True
#
#
# for selection_method in selection_methods:
#     print(f"\n🔍 Selekcja: {selection_method.upper()}")
#
#     def fitness_func(ga_instance, solution, solution_idx):
#         return fitnessFunction(solution, num_vars, bits_per_var, is_minimum)
#
#     # GA do selekcji (1 generacja)
#     ga_sel = pygad.GA(
#         fitness_func=fitness_func,
#         parent_selection_type=selection_method,
#         crossover_type="uniform",  # tymczasowe
#         K_tournament=3,
#         num_generations=1,
#         logger=logging.getLogger(selection_method),
#         **common_args
#     )
#     ga_sel.run()
#
#     # Oblicz fitnessy i wybierz najlepszych
#     fitnesses = [fitness_func(None, ind, i) for i, ind in enumerate(ga_sel.population)]
#     sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
#     selected_population = [
#         [int(round(gene)) for gene in ga_sel.population[i]]
#         for i in sorted_indices[:common_args["num_parents_mating"]]
#     ]
#
#     # Krzyżowanie + Mutacja
#     for crossover_name, crossover_strategy in crossover_methods.items():
#         print(f"  ↪ Krzyżowanie: {crossover_name}")
#         for mutation_name, mutation_strategy in mutation_methods.items():
#             print(f"    ⚙️ Mutacja: {mutation_name}")
#
#             ga_cross = pygad.GA(
#                 fitness_func=fitness_func,
#                 parent_selection_type="tournament",
#                 crossover_type=crossover_strategy,
#                 mutation_type=mutation_strategy,
#                 K_tournament=3,
#                 num_generations=num_generations,
#                 initial_population=selected_population,
#                 logger=logging.getLogger(f"{selection_method}_{crossover_name}_{mutation_name}"),
#                 **common_args
#             )
#
#             ga_cross.run()
#
#             solution, solution_fitness, _ = ga_cross.best_solution()
#
#             key = f"{selection_method}_{crossover_name}_{mutation_name}"
#             experiment_results[key] = {
#                 "fitness": solution_fitness,
#                 "solution": decodeInd(solution, bits_per_var, num_vars),
#                 "fitness_history": [
#                     1.0 / val if is_minimum else val
#                     for val in ga_cross.best_solutions_fitness
#                 ]
#             }
#
# # ===== Wykresy porównawcze =====
# plt.figure(figsize=(12, 6))
# for key, data in experiment_results.items():
#     plt.plot(data["fitness_history"], label=key)
# plt.title("Porównanie selekcji + krzyżowania (fitness)")
# plt.xlabel("Iteracja")
# plt.ylabel("Fitness")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


import logging
import pygad
import numpy as np
import benchmark_functions as bf
from opfunu.cec_based import F132014
import matplotlib.pyplot as plt
import pandas as pd

from crossovers import my_single_point_crossover, my_uniform_crossover, my_discrete_crossover, \
    my_two_point_crossover

# ===== KONFIGURACJA UŻYTKOWNIKA =====
use_binary = True  # <-- Zmienna sterująca: True = binarny, False = rzeczywisty

# ===== Funkcja dekodująca osobnika binarnego =====
def decodeInd(individual, bits_per_var=20, var_count=5, x_min=-32.768, x_max=32.768):
    decoded = []
    for i in range(var_count):
        start = i * bits_per_var
        end = start + bits_per_var
        binary_slice = individual[start:end]
        binary_str = ''.join(str(min(1, max(0, int(round(bit))))) for bit in binary_slice)
        decimal_value = int(binary_str, 2)
        max_decimal = 2**bits_per_var - 1
        real_value = x_min + (decimal_value / max_decimal) * (x_max - x_min)
        decoded.append(real_value)
    return decoded

# ===== Benchmark funkcja =====
def get_function(name, ndim):
    if name == "Hypersphere":
        return bf.Hypersphere(n_dimensions=ndim)
    elif name == "Shifted and Rotated HappyCat Function":
        func = F132014(ndim=ndim)
        return func.evaluate

# ===== Funkcja przystosowania =====
def fitnessFunction(individual, num_vars, bits_per_var, is_minimum=False):
    if use_binary:
        real_values = decodeInd(individual, bits_per_var, num_vars)
    else:
        real_values = individual
    fitness = func(real_values)
    return fitness + 1e-6

# ====== Mutacje ======
def swap_mutation(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        idx1, idx2 = np.random.choice(offspring.shape[1], 2, replace=False)
        offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]
    return offspring

mutation_methods = {
    "random": "random",
    "swap": swap_mutation
}

# ===== Parametry ogólne =====
num_vars = 5
bits_per_var = 20
num_genes = num_vars * bits_per_var if use_binary else num_vars
func = get_function("Hypersphere", num_vars)
num_generations = 50
is_minimum = False

# ===== Parametry GA (stałe) =====
common_args = dict(
    sol_per_pop=80,
    num_parents_mating=50,
    num_genes=num_genes,
    mutation_num_genes=1,
    keep_elitism=1,
    init_range_low=0,
    init_range_high=1,
    random_mutation_max_val=1,
    random_mutation_min_val=0
)

# ===== Eksperyment główny =====
selection_methods = ["tournament", "rws", "random"]
crossover_methods = {
    "single_point": "single_point",
    "two_points": "two_points",
    "uniform": "uniform",
    "my_single_point": my_single_point_crossover,
    "my_two_point": my_two_point_crossover,
    "my_uniform": my_uniform_crossover,
    "my_discrete": my_discrete_crossover
}

experiment_results = {}
#
# for selection_method in selection_methods:
#     print(f"\n🔍 Selekcja: {selection_method.upper()}")
#
#     def fitness_func(ga_instance, solution, solution_idx):
#         return fitnessFunction(solution, num_vars, bits_per_var, is_minimum)
#
#     ga_sel = pygad.GA(
#         fitness_func=fitness_func,
#         parent_selection_type=selection_method,
#         crossover_type="uniform",
#         K_tournament=3,
#         num_generations=1,
#         logger=logging.getLogger(selection_method),
#         **common_args
#     )
#     ga_sel.run()
#
#     fitnesses = [fitness_func(None, ind, i) for i, ind in enumerate(ga_sel.population)]
#     sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
#     selected_population = [
#         [int(round(g)) if use_binary else g for g in ga_sel.population[i]]
#         for i in sorted_indices[:common_args["num_parents_mating"]]
#     ]
#
#     for crossover_name, crossover_strategy in crossover_methods.items():
#         print(f"  ↪ Krzyżowanie: {crossover_name}")
#         for mutation_name, mutation_strategy in mutation_methods.items():
#             print(f"    ⚙️ Mutacja: {mutation_name}")
#
#             ga_cross = pygad.GA(
#                 fitness_func=fitness_func,
#                 parent_selection_type="tournament",
#                 crossover_type=crossover_strategy,
#                 mutation_type=mutation_strategy,
#                 K_tournament=3,
#                 num_generations=num_generations,
#                 initial_population=selected_population,
#                 logger=logging.getLogger(f"{selection_method}_{crossover_name}_{mutation_name}"),
#                 **common_args
#             )
#
#             ga_cross.run()
#
#             solution, solution_fitness, _ = ga_cross.best_solution()
#             solution_decoded = decodeInd(solution, bits_per_var, num_vars) if use_binary else solution
#
#             key = f"{selection_method}_{crossover_name}_{mutation_name}"
#             experiment_results[key] = {
#                 "fitness": solution_fitness,
#                 "solution": solution_decoded,
#                 "fitness_history": [1.0 / val if is_minimum else val for val in ga_cross.best_solutions_fitness]
#             }
def fitness_func(ga_instance, solution, solution_idx):
    return fitnessFunction(solution, num_vars, bits_per_var, is_minimum)

for selection_method in selection_methods:
    for crossover_name, crossover_strategy in crossover_methods.items():
        for mutation_name, mutation_strategy in mutation_methods.items():
            print(f"🔍 {selection_method} | {crossover_name} | {mutation_name}")

            ga = pygad.GA(
                fitness_func=fitness_func,
                parent_selection_type=selection_method,
                crossover_type=crossover_strategy if not callable(crossover_strategy) else None,
                mutation_type=mutation_strategy,
                K_tournament=3,
                num_generations=num_generations,
                logger=logging.getLogger(f"{selection_method}_{crossover_name}_{mutation_name}"),
                **common_args
            )

            ga.run()

            solution, solution_fitness, _ = ga.best_solution()
            solution_decoded = decodeInd(solution, bits_per_var, num_vars) if use_binary else solution

            key = f"{selection_method}_{crossover_name}_{mutation_name}"
            experiment_results[key] = {
                "fitness": solution_fitness,
                "solution": solution_decoded,
                "fitness_history": [1.0 / val if is_minimum else val for val in ga.best_solutions_fitness]
            }


# ===== Czytelniejszy wykres z podziałem i legendą =====
plt.figure(figsize=(14, 7))
for key, data in experiment_results.items():
    plt.plot(data["fitness_history"], label=key, linewidth=1)
plt.title("Porównanie selekcji + krzyżowania + mutacji (f(x))", fontsize=14)
plt.xlabel("Iteracja", fontsize=12)
plt.ylabel("Wartość funkcji celu", fontsize=12)
plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ===== Tabela wyników końcowych =====
results_table = pd.DataFrame([
    {
        "Metoda": key,
        "Najlepszy wynik f(x)": round(data["fitness"], 8),
        "Rozwiązanie": np.round(data["solution"], 4).tolist()
    }
    for key, data in experiment_results.items()
])

# Posortowane rosnąco (bo minimalizacja)
results_table = results_table.sort_values(by="Najlepszy wynik f(x)")

# Wyświetlenie
print("\n📊 Tabela wyników końcowych:\n")
print(results_table.to_string(index=False))