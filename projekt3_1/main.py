import logging
import pygad
import numpy as np
import benchmark_functions as bf
from opfunu.cec_based import F132014
import matplotlib.pyplot as plt
import pandas as pd

from crossovers import my_single_point_crossover, my_uniform_crossover, my_discrete_crossover, \
    my_two_point_crossover
from mutations import swap_mutation, boundary_mutation, one_point_mutation, two_point_mutation
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

    if is_minimum:
        return 1.0 / (fitness + 1e-8)  # minimalizacja przez maksymalizację
    else:
        return fitness


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
    keep_elitism=3, #domyslnie bylo 1
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
mutation_methods = {
    "random": "random",
    "swap": swap_mutation,
    "boundary": boundary_mutation,
    "one_point": one_point_mutation,
    "two_point": two_point_mutation
}

experiment_results = {}

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
                "fitness_history": ga.best_solutions_fitness
            }


# ===== Czytelniejszy wykres z podziałem i legendą =====
for selection_method in selection_methods:
    plt.figure(figsize=(14, 7))
    subset_keys = [k for k in experiment_results.keys() if k.startswith(selection_method)]

    for key in subset_keys:
        data = experiment_results[key]
        history = data["fitness_history"]
        if is_minimum:
            history = [1.0 / (val + 1e-8) for val in history]

        plt.plot(history, label=key, linewidth=1)

    plt.title(f"Selekcja: {selection_method.upper()} — Porównanie krzyżowania + mutacji", fontsize=14)
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