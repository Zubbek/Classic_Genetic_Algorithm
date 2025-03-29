import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from decimal import Decimal, getcontext
import math
import benchmark_functions as bf
from opfunu.cec_based.cec2014 import F12014
from Population import Population

#  potem do dodania jak wybierzemy --------------------------------------------------------------------------------
getcontext().prec = 50 

def get_function(name, ndim):
    """Zwraca wybraną funkcję testową na podstawie jej nazwy."""
    if name == "Hypersphere":
        return bf.Hypersphere(n_dimensions=ndim)
    elif name == "Rotated High Conditioned Elliptic Function":
        func = F12014(ndim=ndim)
        return func.evaluate


# od wyukresów aby sie dało przewijać - jescze do dopracowania------------------------------------------------------------------

class PlotViewer:
    def __init__(self, root, best_fitness_values, avg_fitness_values, std_fitness_values):
        """Klasa do wyświetlania wykresów wyników algorytmu genetycznego."""
        self.plot_window = tk.Toplevel(root)
        self.plot_window.title("Wykresy Algorytmu Genetycznego")
        self.plot_window.geometry("850x600")

        self.figures = []
        self.current_index = 0

        # **🔹 Wykres 1: Najlepsza wartość funkcji celu od iteracji 🔹**
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(range(len(best_fitness_values)), best_fitness_values, marker='o', linestyle='-', color='red')
        ax1.set_xlabel("Iteracja")
        ax1.set_ylabel("Najlepsza wartość funkcji")
        ax1.set_title("Najlepsza wartość funkcji celu w kolejnych iteracjach")
        ax1.grid(True)
        self.figures.append(fig1)

        # **🔹 Wykres 2: Średnia wartość i odchylenie standardowe 🔹**
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

#  tutaj obsłużyc te wszystki ecklasy i funkcje jak wiesz co twoja klasa robi to mozesz dopisać -------------------------------- ---------------  
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
    start_ = get_value(begin_var, -10.0)
    end_ = get_value(end_var, 10.0)
    precision = get_value(precision_var, 4, int)
    population_size = get_value(population_var, 50, int)
    epochs = get_value(epochs_var, 100, int)
    variables_count = get_value(params_var, 10, int)
    elite_percent = get_value(elite_var, 10.0) / 100
    cross_prob = get_value(cross_prob_var, 0.8)
    mutation_prob = get_value(mutation_prob_var, 0.05)
    inversion_prob = get_value(inversion_prob_var, 0.01)
    selection_method = selection_var.get() or "Roulette Wheel"
    best_select_percent = get_value(best_select_var, 20.0) / 100
    tournament_size = get_value(tournament_var, 3, int) if selection_method == "Tournament" else None
    cross_method = cross_method_var.get() or "Single Point"
    mutation_method = mutation_method_var.get() or "One Point"
    function_name = function_var.get() or "Rastrigin"
    is_maximization = maximization_var.get() if maximization_var.get() is not None else True


    func = get_function(function_name, variables_count)
    
    population = Population(variables_count, population_size, precision, start_, end_, func, "max" if is_maximization else "min")
    
    best_fitness_values = []
    avg_fitness_values = []
    std_fitness_values = []

    for _ in range(epochs):
        # Selekcja
        # Selekcja - wybór najlepszych osobników
        if selection_method == "Roulette Wheel":
            selected_individuals = population.getBestByRulet()
            
        elif selection_method == "Tournament":
            selected_individuals = population.getBestByTournament(tournament_size)

        elif selection_method == "Best solution":
            selected_individuals = population.getBestBySelection(best_select_percent)
            
        # population.individuals =selected_individuals
                
        # Krzyżowanie
        population.population_after_single_point_crossover(
            crossover_method_number=1 if cross_method == "Single Point" else 2, 
            crossover_rate=cross_prob
        )

        # Mutacja
        population.population_after_mutationr(mutation_method, mutation_prob)

                
                # Inwersja
        population.inversion(inversion_prob)
        # print(population.individuals)
        
        # Elitaryzm
        elite_individuals = population.elitism(elite_percent)
        population.individuals.extend(elite_individuals)
        
       # Obliczanie wartości funkcji celu dla populacji
        fitness_dict = population.getCell()  # Zwraca słownik {fenotyp: wartość funkcji celu}
        fitness_values = list(fitness_dict.values())  # Pobiera same wartości fitness

        # Obliczanie metryk fitness
        best_fitness_values.append(max(fitness_values))  # Najlepsza wartość fitness
        avg_fitness = sum(fitness_values) / len(fitness_values)  # Średnia wartość fitness
        avg_fitness_values.append(avg_fitness)

        # Odchylenie standardowe fitness
        std_fitness_values.append((sum((x - avg_fitness) ** 2 for x in fitness_values) / len(fitness_values)) ** 0.5)
        
        
        # fitness_values = [float(population.fitness(individual)) for individual in population.individuals]
        # best_fitness_values.append(max(fitness_values))  # Najlepsza wartość fitness
        # avg_fitness_values.append(sum(fitness_values) / len(fitness_values))  # Średnia wartość fitness
        # std_fitness_values.append((sum((x - avg_fitness_values[-1]) ** 2 for x in fitness_values) / len(fitness_values)) ** 0.5)  # Odchylenie standardowe


    # Wyświetlenie wyników na wykresie
    PlotViewer(root, best_fitness_values, avg_fitness_values, std_fitness_values)




# to tam mało istotne co jest niżej ----------------------------------------------------------------------------------------------

def update_tournament_visibility(*args):
    """Pokazuje pole 'Tournament size' tylko jeśli wybrano 'Tournament'."""
    if selection_var.get() == "Tournament":
        tournament_label.grid(row=2, column=0, padx=5, pady=2)
        tournament_entry.grid(row=2, column=1, padx=5, pady=2)
    else:
        tournament_label.grid_remove()
        tournament_entry.grid_remove()
        
# Tworzenie głównego okna
root = tk.Tk()
root.title("Genetic Algorithm Configuration")
root.geometry("350x550")

# Tworzenie pól wejściowych
begin_var = tk.StringVar()
end_var = tk.StringVar()
precision_var = tk.StringVar()
population_var = tk.StringVar()
epochs_var = tk.StringVar()

elite_var = tk.StringVar()
cross_prob_var = tk.StringVar()
mutation_prob_var = tk.StringVar()
inversion_prob_var = tk.StringVar()
best_select_var = tk.StringVar()
tournament_var = tk.StringVar()
params_var = tk.StringVar(value="10")
choices = ["10", "20", "30", "50", "100"]

# Tworzenie list rozwijanych
selection_var = tk.StringVar(value="Roulette Wheel")
selection_var.trace_add("write", update_tournament_visibility)  # Automatyczna reakcja na zmianę

cross_method_var = tk.StringVar(value="One-Point")
mutation_method_var = tk.StringVar(value="Boundary")
function_var = tk.StringVar(value="Hypersphere")

# Minimalizacja/maksymalizacja (radio buttons)
maximization_var = tk.BooleanVar(value=True)

# Tworzymy ramkę dla lepszego układu
frame = tk.Frame(root)
frame.pack(pady=12)

fields = [
    ("Begin of the range", begin_var),
    ("End of the range", end_var),
    ("Precision", precision_var),
    ("Population", population_var),
    ("Epochs", epochs_var),
    # ("Number of parameters", params_var),
    ("Percentage elite strategy", elite_var),
    ("Cross probability", cross_prob_var),
    ("Mutation probability", mutation_prob_var),
    ("Inversion probability", inversion_prob_var),
]

# Tworzenie pól w układzie tabeli (grid)
for i, (label, var) in enumerate(fields):
    tk.Label(frame, text=label, anchor="w", width=23).grid(row=i, column=0, padx=0, pady=2)
    tk.Entry(frame, textvariable=var, width=15).grid(row=i, column=1, padx=5, pady=2)

# Listy rozwijane
tk.Label(frame, text="Number of parameters", anchor="w", width=23).grid(row=11, column=0, padx=0, pady=2)
dropdown = tk.OptionMenu(frame, params_var, *choices)
dropdown.grid(row=11, column=1)

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

select_frame = tk.Frame(root)
select_frame.pack(pady=0)

tk.Label(select_frame, text="Cross method", anchor="w", width=25).grid(row=0, column=0)
cross_method_combobox = ttk.Combobox(select_frame, textvariable=cross_method_var, values=["One-Point", "Two-Point", "Tree-Point", "Uniform crossover", "Granular crossover"], width=15)
cross_method_combobox.grid(row=1, column=1)
cross_method_combobox.state(["readonly"])

tk.Label(select_frame, text="Mutation method", anchor="w", width=25).grid(row=1, column=0)
mutation_method_combobox = ttk.Combobox(select_frame, textvariable=mutation_method_var, values=["Boundary", "One-Point", "Two-Point"], width=15)
mutation_method_combobox.grid(row=2, column=1)
mutation_method_combobox.state(["readonly"]) 

tk.Label(select_frame, text="Function to calculation", anchor="w", width=25).grid(row=2, column=0)
function_combobox = ttk.Combobox(select_frame, textvariable=function_var, values=["Hypersphere", "Rotated High Conditioned Elliptic Function"], width=15)
function_combobox.grid(row=3, column=1)
function_combobox.state(["readonly"])

# Radio buttony dla minimalizacji/maksymalizacji
radio_frame = tk.Frame(root)
radio_frame.pack(pady=2)

tk.Label(radio_frame, text="Optimization type:", anchor="w", width=25).grid(row=0, column=0)
tk.Radiobutton(radio_frame, text="Minimization", variable=maximization_var, value=False).grid(row=1, column=0)
tk.Radiobutton(radio_frame, text="Maximization", variable=maximization_var, value=True).grid(row=1, column=1)

# Przycisk startowy
tk.Button(root, text="Start", command=start_algorithm, width=7, height=1).pack(pady=10)

# Uruchomienie aplikacji
root.mainloop()
