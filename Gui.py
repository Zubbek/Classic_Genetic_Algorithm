import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import benchmark_functions as bf
# from opfunu.cec_based.cec2014 import F132014
from Population import Population
import csv

def get_function(name, ndim):
    """Zwraca wybraną funkcję testową na podstawie jej nazwy."""
    if name == "Hypersphere":
        return bf.Hypersphere(n_dimensions=ndim)
    elif name == "Shifted and Rotated HappyCat Function":
        return bf.Hypersphere(n_dimensions=ndim)

        # func = F132014(ndim=ndim)
        # return func.evaluate

def save_results_csv(filename, best_fitness_values, avg_fitness_values, std_fitness_values):
    """Zapisuje wyniki algorytmu do pliku CSV."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iteracja", "Najlepsza wartosc", "Srednia wartosc", "Odchylenie standardowe"])
        for i, (best, avg, std) in enumerate(zip(best_fitness_values, avg_fitness_values, std_fitness_values)):
            writer.writerow([i, best, avg, std])
    print(f"Wyniki zapisano do pliku: {filename}")

class PlotViewer:
    def __init__(self, root, best_fitness_values, avg_fitness_values, std_fitness_values):
        """Klasa do wyświetlania wykresów wyników algorytmu genetycznego."""
        self.plot_window = tk.Toplevel(root)
        self.plot_window.title("Wykresy Algorytmu Genetycznego")
        self.plot_window.geometry("850x600")

        self.figures = []
        self.current_index = 0

        # **🔹 Wykres 1: Najlepsza wartość funkcji celu 🔹**
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(range(len(best_fitness_values)), best_fitness_values, marker='o', linestyle='-', color='red')
        ax1.set_xlabel("Iteracja")
        ax1.set_ylabel("Najlepsza wartość funkcji")
        ax1.set_title("Najlepsza wartość funkcji celu w kolejnych iteracjach")
        ax1.grid(True)
        self.figures.append(fig1)

        # **🔹 Wykres 2: Średnia wartość i odchylenie standardowe (obszar) 🔹**
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
        self.figures.append(fig2)

        # **🔹 Wykres 3: Średnia wartość funkcji celu 🔹**
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(range(len(avg_fitness_values)), avg_fitness_values, marker='o', linestyle='-', color='green')
        ax3.set_xlabel("Iteracja")
        ax3.set_ylabel("Średnia wartość funkcji")
        ax3.set_title("Średnia wartość funkcji celu w kolejnych iteracjach")
        ax3.grid(True)
        self.figures.append(fig3)

        # **🔹 Wykres 4: Odchylenie standardowe jako wykres liniowy 🔹**
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(range(len(std_fitness_values)), std_fitness_values, marker='o', linestyle='-', color='purple')
        ax4.set_xlabel("Iteracja")
        ax4.set_ylabel("Odchylenie standardowe")
        ax4.set_title("Odchylenie standardowe w kolejnych iteracjach")
        ax4.grid(True)
        self.figures.append(fig4)

        # **🔹 Wyświetlanie wykresu 🔹**
        self.canvas = FigureCanvasTkAgg(self.figures[self.current_index], master=self.plot_window)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.pack(pady=20)

        # **🔹 Kontrolki nawigacji 🔹**
        self.button_frame = tk.Frame(self.plot_window)
        self.button_frame.pack(pady=10)

        self.prev_button = ttk.Button(self.button_frame, text="◀ Poprzedni", command=self.show_previous)
        self.prev_button.grid(row=0, column=0, padx=10)

        self.next_button = ttk.Button(self.button_frame, text="Następny ▶", command=self.show_next)
        self.next_button.grid(row=0, column=1, padx=10)

        self.update_buttons()

    def show_previous(self):
        """Przełącza na poprzedni wykres."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()

    def show_next(self):
        """Przełącza na następny wykres."""
        if self.current_index < len(self.figures) - 1:
            self.current_index += 1
            self.update_plot()

    def update_plot(self):
        """Aktualizuje wyświetlany wykres."""
        self.plot_widget.destroy()
        self.canvas = FigureCanvasTkAgg(self.figures[self.current_index], master=self.plot_window)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.pack(pady=20)
        self.update_buttons()

    def update_buttons(self):
        """Aktualizuje stan przycisków nawigacyjnych."""
        self.prev_button.config(state="normal" if self.current_index > 0 else "disabled")
        self.next_button.config(state="normal" if self.current_index < len(self.figures) - 1 else "disabled")

def start_algorithm():
    """Uruchamia algorytm genetyczny z wartościami z GUI."""
    def get_value(var, default, cast_type=float):
        """Funkcja pomocnicza pobierająca wartości z GUI, z obsługą błędów."""
        try:
            value = var.get()
            return cast_type(value) if value else default
        except ValueError:
            return default

    # Pobieranie wartości z GUI lub użycie domyślnych
    start_ = get_value(begin_var, -50.0)
    end_ = get_value(end_var, 50.0)
    precision = get_value(precision_var, 4, int)
    population_size = get_value(population_var, 50, int)
    epochs = get_value(epochs_var, 500, int)
    variables_count = get_value(params_var, 10, int)
    elite_percent = get_value(elite_percent_var, 10.0)/100
    elite_count = int(get_value(elite_count_var, 0, int)) if elite_count_var.get() else None
    cross_prob = get_value(cross_prob_var, 0.8)
    mutation_prob = get_value(mutation_prob_var, 0.05)
    inversion_prob = get_value(inversion_prob_var, 0.01)
    std_dev = get_value(inversion_prob_var, 0.01) # This seems to be a typo, should likely be a different variable or a fixed value
    alpha = get_value(alpha_var, 0.01) # Pobranie alpha
    beta = get_value(beta_var, 0.01)   # Pobranie beta
    selection_method = selection_var.get() or "Roulette Wheel"
    best_select_percent = get_value(best_select_var, 20.0)
    tournament_size = get_value(tournament_var, 3, int) if selection_method == "Tournament" else None
    cross_method = cross_method_var.get() or "Single Point"
    cross_probability = get_value(cross_probability_var, 0.7) if cross_method == "Uniform crossover" else None

    mutation_method = mutation_method_var.get() or "One Point"
    function_name = function_var.get() or "Rastrigin"
    is_maximization = maximization_var.get() if maximization_var.get() is not None else True
    representation_type = "real" if type_var.get() else "binary"

    
    print(representation_type)
    crossover_mapping = {
            "One-Point": 1,
            "Two-Point": 2,
            "Uniform crossover": 3,
            "Granular crossover": 4,
            "arithmetic_crossover_real": 5,
            "linear_crossover_real": 6,
            "blend_crossover_alpha_real": 7,
            "blend_crossover_alpha_beta_real": 8,
            "average_crossover_real": 9,
    }

    func = get_function(function_name, variables_count)

    population = Population(variables_count, population_size, precision, start_, end_, func, "max" if is_maximization else "min", std_dev, alpha, beta, representation_type)

    best_fitness_values = []
    avg_fitness_values = []
    std_fitness_values = []
    elite_individuals = []
    # Zarejestruj czas początkowy
    start_time = time.time()

    for _ in range(epochs):

        # Elitaryzm
        elite_individuals=population.elitism(elite_percent,elite_count)

        # Selekcja
        # Selekcja - wybór najlepszych osobników
        if selection_method == "Roulette Wheel":
            population.getBestByRulet(best_select_percent)

        elif selection_method == "Tournament":
            population.getBestByTournament(tournament_size)

        elif selection_method == "Best solution":
            population.getBestBySelection(best_select_percent)


        # Krzyżowanie
        crossover_method_number = crossover_mapping.get(cross_method, 1)

        # Wywołanie funkcji
        population.population_after_crossover(
            crossover_method_number=crossover_method_number,
            crossover_rate=cross_prob,
            elite=len(elite_individuals),
            cross_probability=cross_probability
        )

        # Mutacja
        population.population_after_mutationr(mutation_method, mutation_prob)


        # Inwersja
        if representation_type == "binary":
            population.inversion(inversion_prob)

        fitness_values = [float(population.fitness(individual)) for individual in population.individuals]
        if is_maximization:
            best_fitness_values.append(max(fitness_values))  # Najlepsza wartość fitness
        else:
            best_fitness_values.append(min(fitness_values))
        avg_fitness_values.append(sum(fitness_values) / len(fitness_values))  # Średnia wartość fitness
        std_fitness_values.append((sum((x - avg_fitness_values[-1]) ** 2 for x in fitness_values) / len(fitness_values)) ** 0.5)  # Odchylenie standardowe

        population.individuals=population.best_individuals
        population.individuals.extend(elite_individuals)

    # Zarejestruj czas zakończenia
    end_time = time.time()
    save_results_csv("wyniki.csv", best_fitness_values, avg_fitness_values, std_fitness_values)
    # Oblicz czas działania algorytmu
    elapsed_time = end_time - start_time
    # Wyświetlenie wyników na wykresie
    plot_viewer = PlotViewer(root, best_fitness_values, avg_fitness_values, std_fitness_values)

    # Dodaj etykietę z czasem wykonania pod wykresami
    time_label = tk.Label(plot_viewer.plot_window, text=f"Czas działania algorytmu: {elapsed_time:.4f} sekundy", font=("Arial", 12))
    # time_label = tk.Label(plot_viewer.plot_window, text=f"Najlepszy wynik: {best_fitness_values[-1]:.4f}", font=("Arial", 12))
    time_label.pack(pady=10)


def update_tournament_visibility(*args):
    if selection_var.get() == "Tournament":
        tournament_label.grid(row=2, column=0, padx=5, pady=2)
        tournament_entry.grid(row=2, column=1, padx=5, pady=2)
    else:
        tournament_label.grid_remove()
        tournament_entry.grid_remove()

def update_ross_probability_visibility(*args):
    if cross_method_var.get() == "Uniform crossover":
        cross_probability_label.grid(row=2, column=0, padx=5, pady=2)
        cross_probability_entry.grid(row=2, column=1, padx=5, pady=2)
    else:
        cross_probability_label.grid_remove()
        cross_probability_entry.grid_remove()

def update_alpha(*args):
    if cross_method_var.get() in ["arithmetic_crossover_real", "blend_crossover_alpha_real", "blend_crossover_alpha_beta_real"]:
        cross_probability_label3.grid(row=2, column=0, padx=5, pady=2)
        cross_probability_entry3.grid(row=2, column=1, padx=5, pady=2)
    else:
        cross_probability_label3.grid_remove()
        cross_probability_entry3.grid_remove()

def update_beta(*args):
    if cross_method_var.get() == "blend_crossover_alpha_beta_real":
        cross_probability_label2.grid(row=3, column=0, padx=5, pady=2)
        cross_probability_entry2.grid(row=3, column=1, padx=5, pady=2)
    else:
        cross_probability_label2.grid_remove()
        cross_probability_entry2.grid_remove()

def update_combobox_options(*args):
    if type_var.get():  # Real
        cross_method_combobox['values'] = [
            "arithmetic_crossover_real", "linear_crossover_real",
            "blend_crossover_alpha_real", "blend_crossover_alpha_beta_real",
            "average_crossover_real"
        ]
        cross_method_var.set("arithmetic_crossover_real")

        mutation_method_combobox['values'] = ["Gaussian", "Uniform"]
        mutation_method_var.set("Gaussian")
    else:  # Binary
        cross_method_combobox['values'] = [
            "One-Point", "Two-Point", "Uniform crossover", "Granular crossover"
        ]
        cross_method_var.set("One-Point")

        mutation_method_combobox['values'] = ["Boundary", "One-Point", "Two-Point"]
        mutation_method_var.set("Boundary")

# Główne okno
root = tk.Tk()
root.title("Genetic Algorithm Configuration")
root.geometry("350x620")

# Zmienne
begin_var = tk.StringVar(value=-10)
end_var = tk.StringVar(value=10)
precision_var = tk.StringVar(value=4)
population_var = tk.StringVar(value=50)
epochs_var = tk.StringVar(value=100)

alpha_var = tk.StringVar(value=0.01)
beta_var = tk.StringVar(value=0.01)

elite_percent_var = tk.StringVar(value=10.0)
elite_count_var = tk.StringVar()

cross_prob_var = tk.StringVar(value=0.8)
mutation_prob_var = tk.StringVar(value=0.05)
inversion_prob_var = tk.StringVar(value=0.01)
best_select_var = tk.StringVar(value=20.0)

tournament_var = tk.StringVar(value=3)
cross_probability_var = tk.StringVar(value=0.7)
params_var = tk.StringVar(value="10")

selection_var = tk.StringVar(value="Roulette Wheel")
selection_var.trace_add("write", update_tournament_visibility)

cross_method_var = tk.StringVar(value="arithmetic_crossover_real")
cross_method_var.trace_add("write", update_ross_probability_visibility)
cross_method_var.trace_add("write", update_alpha)
cross_method_var.trace_add("write", update_beta)

mutation_method_var = tk.StringVar(value="Gaussian")
function_var = tk.StringVar(value="Hypersphere")

maximization_var = tk.BooleanVar(value=True)
type_var = tk.BooleanVar(value=True)
type_var.trace_add("write", update_combobox_options)

# UI: Typ zmiennych (Binary/Real)
radio_frame1 = tk.Frame(root)
radio_frame1.pack(pady=2)

tk.Label(radio_frame1, text=" type:", anchor="w", width=25).grid(row=0, column=0)
tk.Radiobutton(radio_frame1, text="Binary", variable=type_var, value=False).grid(row=1, column=0)
tk.Radiobutton(radio_frame1, text="Real", variable=type_var, value=True).grid(row=1, column=1)

# Dane podstawowe
frame = tk.Frame(root)
frame.pack(pady=12)

fields = [
    ("Begin of the range", begin_var),
    ("End of the range", end_var),
    ("Precision", precision_var),
    ("Population", population_var),
    ("Epochs", epochs_var),
    ("Number of parameters", params_var),
    ("Percentage elite strategy", elite_percent_var),
    ("Count elite strategy", elite_count_var),
    ("Mutation probability", mutation_prob_var),
    ("Inversion probability", inversion_prob_var),
]

for i, (label, var) in enumerate(fields):
    tk.Label(frame, text=label, anchor="w", width=23).grid(row=i, column=0, padx=0, pady=2)
    tk.Entry(frame, textvariable=var, width=15).grid(row=i, column=1, padx=5, pady=2)

# Selection
select_frame1 = tk.Frame(root)
select_frame1.pack(pady=0)

tk.Label(select_frame1, text="Selection method", anchor="w", width=25).grid(row=0, column=0)
selection_combobox = ttk.Combobox(select_frame1, textvariable=selection_var, values=["Roulette Wheel", "Best solution", "Tournament"], width=15)
selection_combobox.grid(row=0, column=1)
selection_combobox.state(["readonly"])

tk.Label(select_frame1, text="Percentage the best to select", anchor="w", width=23).grid(row=1, column=0, padx=0, pady=2)
tk.Entry(select_frame1, textvariable=best_select_var, width=15).grid(row=1, column=1, padx=5, pady=2)

tournament_label = tk.Label(select_frame1, text="Tournament size", anchor="w", width=25)
tournament_entry = tk.Entry(select_frame1, textvariable=tournament_var, width=15)

update_tournament_visibility()

# Crossover
select_frame2 = tk.Frame(root)
select_frame2.pack(pady=0)

tk.Label(select_frame2, text="Cross method", anchor="w", width=25).grid(row=0, column=0)
cross_method_combobox = ttk.Combobox(select_frame2, textvariable=cross_method_var, values=[], width=15)
cross_method_combobox.grid(row=0, column=1)
cross_method_combobox.state(["readonly"])

tk.Label(select_frame2, text="Cross probability", anchor="w", width=23).grid(row=1, column=0, padx=0, pady=2)
tk.Entry(select_frame2, textvariable=cross_prob_var, width=15).grid(row=1, column=1, padx=5, pady=2)

cross_probability_label = tk.Label(select_frame2, text="cross probability", anchor="w", width=25)
cross_probability_entry = tk.Entry(select_frame2, textvariable=cross_probability_var, width=15)

cross_probability_label3 = tk.Label(select_frame2, text="Alpha", anchor="w", width=25)
cross_probability_entry3 = tk.Entry(select_frame2, textvariable=alpha_var, width=15)

cross_probability_label2 = tk.Label(select_frame2, text="Beta", anchor="w", width=25)
cross_probability_entry2 = tk.Entry(select_frame2, textvariable=beta_var, width=15)

update_ross_probability_visibility()
update_alpha()
update_beta()

# Mutation & Function
select_frame = tk.Frame(root)
select_frame.pack(pady=0)

tk.Label(select_frame, text="Mutation method", anchor="w", width=25).grid(row=1, column=0)
mutation_method_combobox = ttk.Combobox(select_frame, textvariable=mutation_method_var, values=[], width=15)
mutation_method_combobox.grid(row=1, column=1)
mutation_method_combobox.state(["readonly"])

tk.Label(select_frame, text="Function to calculation", anchor="w", width=25).grid(row=2, column=0)
function_combobox = ttk.Combobox(select_frame, textvariable=function_var, values=["Hypersphere", "Shifted and Rotated HappyCat Function"], width=15)
function_combobox.grid(row=2, column=1)
function_combobox.state(["readonly"])

# Maximization / Minimization
radio_frame = tk.Frame(root)
radio_frame.pack(pady=2)

tk.Label(radio_frame, text="Optimization type:", anchor="w", width=25).grid(row=0, column=0)
tk.Radiobutton(radio_frame, text="Minimization", variable=maximization_var, value=False).grid(row=1, column=0)
tk.Radiobutton(radio_frame, text="Maximization", variable=maximization_var, value=True).grid(row=1, column=1)

# Start button
tk.Button(root, text="Start", command=start_algorithm, width=7, height=1).pack(pady=10)

# ⬇️ TERAZ dopiero aktualizacja comboboxów
update_combobox_options()

# Start GUI
root.mainloop()
