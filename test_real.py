import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

import benchmark_functions as bf
from opfunu.cec_based.cec2014 import F132014
from Population import Population

function_version = "Shifted_and_Rotated_HappyCat_Function_real"

# Funkcja zwracająca wybraną funkcję testową na podstawie nazwy
def get_function(name, ndim):
    if name == "Hypersphere":
        return bf.Hypersphere(n_dimensions=ndim)
    elif name == "Shifted and Rotated HappyCat Function":
        func = F132014(ndim=ndim)
        return func.evaluate

# Funkcja zapisująca dane eksperymentalne do pliku CSV (dodaje wiersz)
def append_results_csv(filename, row):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

# Funkcja wykonująca pojedynczy eksperyment dla reprezentacji rzeczywistej
def run_experiment_real(selection_method, cross_method, mutation_method, run_number):
    # Parametry domyślne
    start_ = -50.0
    end_ = 50.0
    precision = 4
    population_size = 50
    epochs = 500
    variables_count = 10
    elite_percent = 10.0/100
    elite_count = None
    cross_prob = 0.8
    mutation_prob = 0.05
    inversion_prob = 0.01 # Inwersja nie jest typowa dla reprezentacji rzeczywistej, ale zostawiam na razie
    best_select_percent = 20.0
    tournament_size = 3  # Używane, gdy selection_method == "Tournament"
    cross_probability = 0.8 # używane dla "Uniform crossover" dla real
    alpha = 0.1
    beta = 0.9
    function_name = "Shifted and Rotated HappyCat Function"  # domyślna funkcja
    is_maximization = True  # optymalizacja: minimalizacja
    representation_type = "real"

    # Mapa metod krzyżowania dla liczb rzeczywistych
    crossover_mapping = {
        "arithmetic_crossover_real": 5,
        "linear_crossover_real": 6,
        "blend_crossover_alpha_real": 7,
        "blend_crossover_alpha_beta_real": 8,
        "average_crossover_real": 9,
    }
    crossover_method_number = crossover_mapping.get(cross_method, 5)

    # Przygotowanie funkcji celu
    func = get_function(function_name, variables_count)

    # Inicjalizacja populacji
    population = Population(variables_count, population_size, precision, start_, end_, func, "max" if is_maximization else "min", std_dev=0.1, alpha=alpha, beta=beta, representation_type=representation_type)

    best_fitness_values = []
    avg_fitness_values = []
    std_fitness_values = []
    elite_individuals = []

    start_time = time.time()

    for _ in range(epochs):
        # Elitaryzm
        elite_individuals = population.elitism(elite_percent, elite_count)

        # Selekcja w zależności od metody
        if selection_method == "Roulette Wheel":
            population.getBestByRulet(best_select_percent)
        elif selection_method == "Tournament":
            population.getBestByTournament(tournament_size)
        elif selection_method == "Best solution":
            population.getBestBySelection(best_select_percent)

        # Krzyżowanie
        population.population_after_crossover(
            crossover_method_number=crossover_method_number,
            crossover_rate=cross_prob,
            elite=len(elite_individuals),
            cross_probability=cross_probability if cross_method == "Uniform crossover" else None # Uniform crossover nie jest typowy dla real, ale zostawiam
        )

        # Mutacja
        population.population_after_mutationr(mutation_method, mutation_prob)

        # Inwersja (może nie mieć sensu dla real, ale zostawiam)
        population.inversion(inversion_prob)

        # Obliczenie statystyk
        fitness_values = [float(population.fitness(individual)) for individual in population.individuals]
        if is_maximization:
            best_fitness = max(fitness_values)
        else:
            best_fitness = min(fitness_values)
        best_fitness_values.append(best_fitness)
        avg = sum(fitness_values) / len(fitness_values)
        avg_fitness_values.append(avg)
        std = (sum((x - avg) ** 2 for x in fitness_values) / len(fitness_values)) ** 0.5
        std_fitness_values.append(std)

        # Utrzymanie najlepszych osobników oraz elitarnych
        population.individuals = population.best_individuals
        population.individuals.extend(elite_individuals)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Wykres 1: Najlepsza wartość funkcji celu
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(range(len(best_fitness_values)), best_fitness_values, marker='o', linestyle='-', color='red')
    ax1.set_xlabel("Iteracja")
    ax1.set_ylabel("Najlepsza wartość funkcji")
    ax1.set_title("Najlepsza wartość funkcji celu w kolejnych iteracjach")
    ax1.grid(True)

    # Wykres 2: Średnia wartość i odchylenie standardowe
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(range(len(avg_fitness_values)), avg_fitness_values, label="Średnia wartość", marker='o', linestyle='-', color='blue')
    ax2.fill_between(range(len(avg_fitness_values)),
                     np.array(avg_fitness_values) - np.array(std_fitness_values),
                     np.array(avg_fitness_values) + np.array(std_fitness_values),
                     color='blue', alpha=0.2, label="Odchylenie standardowe")
    ax2.set_xlabel("Iteracja")
    ax2.set_ylabel("Wartość funkcji")
    ax2.set_title("Średnia wartość funkcji celu i odchylenie standardowe")
    ax2.legend()
    ax2.grid(True)

    # Przygotowanie nazw plików
    sel_str = selection_method.replace(" ", "")
    cross_str = cross_method.replace(" ", "")
    mut_str = mutation_method.replace(" ", "")
    file_chart1 = f"{function_version}_{sel_str}_{cross_str}_{mut_str}_{run_number}_1.png"
    file_chart2 = f"{function_version}_{sel_str}_{cross_str}_{mut_str}_{run_number}_2.png"

    fig1.savefig(file_chart1)
    fig2.savefig(file_chart2)
    plt.close(fig1)
    plt.close(fig2)

    return elapsed_time, best_fitness_values[-1], file_chart1, file_chart2

def main_real():
    # Kombinacje metod dla liczb rzeczywistych
    selection_methods = ["Roulette Wheel", "Best solution", "Tournament"]
    cross_methods = ["arithmetic_crossover_real", "linear_crossover_real", "blend_crossover_alpha_real", "blend_crossover_alpha_beta_real", "average_crossover_real"]
    mutation_methods = ["Gaussian", "Uniform"]

    results_csv = "test_results_real.csv"
    summery_csv = "test_summery_real.csv"

    # Zapis nagłówka do pliku z wynikami eksperymentów
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Selection Method", "Cross Method", "Mutation Method", "Run", "Time (s)", "Best Fitness", "Chart1", "Chart2"])

    # Zapis nagłówka do pliku z podsumowaniami (rozszerzony)
    with open(summery_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Selection Method", "Cross Method", "Mutation Method",
                            "Średni czas", "Średni Fitness",
                            "Najlepszy Fitness", "Najgorszy Fitness",
                            "Najlepszy czas", "Najgorszy czas"])

    # Iteracja przez wszystkie kombinacje
    for sel in selection_methods:
        for cross in cross_methods:
            for mut in mutation_methods:
                run_times = []
                run_fitnesses = []
                for i in range(1, 11):
                    print(f"Uruchomienie REAL: {sel} | {cross} | {mut} | Run {i}")
                    elapsed_time, best_fit, chart1, chart2 = run_experiment_real(sel, cross, mut, i)
                    run_times.append(elapsed_time)
                    run_fitnesses.append(best_fit)
                    # Zapis pojedynczego wyniku do pliku wynikowego
                    row = [sel, cross, mut, i, elapsed_time, best_fit, chart1, chart2]
                    append_results_csv(results_csv, row)
                # Obliczenie średnich dla 10 powtórzeń
                avg_time = round(sum(run_times) / len(run_times),4)
                avg_fitness = round(sum(run_fitnesses) / len(run_fitnesses),4)
                best_run_time = round(min(run_times),4)
                worst_run_time = round(max(run_times),4)
                # Przyjmując, że optymalizujemy przez minimalizację, najlepszy fitness = minimalny
                best_run_fitness = round(min(run_fitnesses),4)
                worst_run_fitness = round(max(run_fitnesses),4)
                # Zapis podsumowania do osobnego pliku CSV
                append_results_csv(summery_csv, [sel, cross, mut, avg_time, avg_fitness,
                                                    best_run_fitness, worst_run_fitness,
                                                    best_run_time, worst_run_time])
    print("Wszystkie wyniki dla reprezentacji rzeczywistej zapisano do plików test_results_real.csv oraz test_summery_real.csv")

if __name__ == "__main__":
    main_real()