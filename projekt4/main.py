from mealpy import FOA
from mealpy.utils.problem import Problem
from mealpy.utils.space import FloatVar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import benchmark_functions as bf
from opfunu.cec_based import F132014

benchmark_configs = [
    {
        "name": "Hypersphere",
        "num_vars": 10,
        "lb": -5,
        "ub": 5,
        "is_minimum": False,
    },
    {
        "name": "Shifted and Rotated HappyCat Function",
        "num_vars": 10,
        "lb": -50,
        "ub": 50,
        "is_minimum": False,
    }
]

pop_sizes = [50]
epochs_list = [1000]
num_runs = 5

all_experiment_results = {}

for config in benchmark_configs:
    func_name = config["name"]
    num_vars = config["num_vars"]
    lb = config["lb"]
    ub = config["ub"]
    is_minimum_problem = config["is_minimum"]

    print(f"\n===== Running experiments for: {func_name} (Dimensions: {num_vars}, Range: [{lb}, {ub}]) =====")

    if func_name == "Hypersphere":
        obj_func = bf.Hypersphere(n_dimensions=num_vars)
    elif func_name == "Shifted and Rotated HappyCat Function":
        happycat_func_instance = F132014(ndim=num_vars)
        obj_func = happycat_func_instance.evaluate
    else:
        raise ValueError(f"Unknown function name: {func_name}")

    problem_dict = {
        "obj_func": obj_func,
        "lb": [lb] * num_vars,
        "ub": [ub] * num_vars,
        "minmax": "min" if is_minimum_problem else "max",
        "bounds": [FloatVar(lb, ub)] * num_vars,
    }
    problem = Problem(**problem_dict)

    current_func_results = {}

    for pop_size in pop_sizes:
        for epochs in epochs_list:
            config_key = (func_name, pop_size, epochs)
            print(f"\nRunning FOA | Pop: {pop_size}, Epochs: {epochs} for {num_runs} runs...")

            all_final_fitness_for_config = []
            all_histories_for_config = []

            for run in range(num_runs):
                model = FOA.OriginalFOA(epoch=epochs, pop_size=pop_size)
                best_agent = model.solve(problem)

                final_fitness = best_agent.target.fitness if hasattr(best_agent.target, 'fitness') else best_agent.target

                all_final_fitness_for_config.append(final_fitness)
                all_histories_for_config.append(model.history.list_global_best_fit)

                print(f"  Run {run+1}/{num_runs}: Best f(x) = {final_fitness:.8f}")

            mean_fitness = np.mean(all_final_fitness_for_config)
            std_fitness = np.std(all_final_fitness_for_config)

            best_final_fitness_overall = np.min(all_final_fitness_for_config) if is_minimum_problem else np.max(all_final_fitness_for_config)

            current_func_results[config_key] = {
                "mean_final_fitness": mean_fitness,
                "std_final_fitness": std_fitness,
                "best_final_fitness": best_final_fitness_overall,
                "all_final_fitness": all_final_fitness_for_config,
                "all_histories": all_histories_for_config
            }
    all_experiment_results[func_name] = current_func_results

for func_name, results_data in all_experiment_results.items():
    is_min_problem_for_func = next(item["is_minimum"] for item in benchmark_configs if item["name"] == func_name)

    table_data = []
    for (f_name, pop_size, epochs), data in results_data.items():
        table_data.append({
            "Algorytm": "FOA.OriginalFOA",
            "Pop size": pop_size,
            "Epochs": epochs,
            "Średnia końcowa fitness": round(data["mean_final_fitness"], 8),
            "Odchylenie std końcowej fitness": round(data["std_final_fitness"], 8),
            "Najlepsza końcowa fitness": round(data["best_final_fitness"], 8)
        })

    df_func_results = pd.DataFrame(table_data)
    df_func_results = df_func_results.sort_values(by="Najlepsza końcowa fitness", ascending=is_min_problem_for_func)

    print(f"\n📊 Wyniki FOA (OriginalFOA) dla funkcji: {func_name}\n")
    print(df_func_results.to_string(index=False))

    # ===== WYKRESY =====
    # === Ewolucja najlepszej wartości funkcji celu ===
    plt.figure(figsize=(12, 7))
    for (f_name, pop_size, epochs), data in results_data.items():
        if data["all_histories"]:
            plt.plot(data["all_histories"][0], label=f"Pop: {pop_size}, Epochs: {epochs}")

    plt.xlabel("Epoka")
    plt.ylabel("Najlepsza wartość funkcji celu")
    plt.title(f"Ewolucja najlepszej wartości funkcji celu na {func_name}")
    plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Średnia wartość funkcji celu i odchylenie standardowe w czasie ===
    plt.figure(figsize=(12, 7))
    for (f_name, pop_size, epochs), data in results_data.items():
        if data["all_histories"]:
            mean_history = np.mean(np.array(data["all_histories"]), axis=0)
            std_history = np.std(np.array(data["all_histories"]), axis=0)
            epochs_range = np.arange(len(mean_history))
            plt.plot(epochs_range, mean_history, label=f"Średnia (Pop: {pop_size}, Epochs: {epochs})", marker='o', markersize=5)
            plt.fill_between(epochs_range, mean_history - std_history, mean_history + std_history, alpha=0.2, label=f"Odchylenie std (Pop: {pop_size}, Epochs: {epochs})")

    plt.xlabel("Epoka")
    plt.ylabel("Wartość funkcji celu")
    plt.title(f"Średnia i odchylenie standardowe wartości funkcji celu na {func_name}")
    plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()