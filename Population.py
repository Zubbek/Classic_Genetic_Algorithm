import random

import numpy as np
from Individual import Individual

class Population:
    def __init__(self, variables_count, population_size, precision, start_, end_, func, optimum, std_dev, alpha, beta, representation_type="binary"):
        self.variables_count = variables_count
        self.population_size = population_size
        self.func = func
        self.representation_type = representation_type
        self.individuals = [Individual(precision, variables_count, start_, end_, self.representation_type) for _ in range(self.population_size)]
        self.optimum = 0 if optimum == "min" else 1
        self.precision = precision
        self.start_ = start_
        self.end_ = end_
        self.std_dev = std_dev
        self.alpha = alpha
        self.beta = beta
        self.best_individuals = []

    def fitness(self, individual: Individual):
        return self.func(individual.chromosom.decode())        

    def getBestBySelection(self, percentage: float):
        """Zwraca najlepsze osobniki jako listę obiektów Individual."""
        size = int(percentage * self.population_size / 100)
        self.best_individuals = sorted(
            self.individuals,
            key=self.fitness,
            reverse=self.optimum
        )[:size]

    def getBestByTournament(self, k):
        """Zwraca najlepsze osobniki według turnieju jako listę obiektów Individual."""
        if k > self.population_size:
            raise ValueError("k parameter cannot be greater than population_size!")

        self.best_individuals = []

        for _ in range(k):
            tournament = random.sample(self.individuals, int(self.population_size / k))  # Losujemy podzbiór

            best = max(tournament, key=self.fitness) if self.optimum == 1 else min(tournament, key=self.fitness)

            self.best_individuals.append(best)  # Teraz przechowujemy cały obiekt Individual            

    def getBestByRulet(self, percentage: float):
      """Zwraca najlepsze osobniki według metody ruletki jako listę obiektów Individual."""
      size = int(percentage * self.population_size / 100)  # Obliczamy liczbę osobników do wybrania
      self.best_individuals = []

      cell = {ind: self.fitness(ind) for ind in self.individuals}

      if self.optimum == 0:
          cell = {key: 1 / value for key, value in cell.items()}

      min_value = min(cell.values())

      if min_value < 0:
          shift_constant = abs(min_value) + 1
          cell = {key: value + shift_constant for key, value in cell.items()}

      total_fitness = sum(cell.values())
      if total_fitness == 0:
          raise ValueError("Total fitness is zero!")

      distribution = {}
      distribution_value = 0

      for individual, fitness in cell.items():
          probability = fitness / total_fitness
          distribution_value += probability
          distribution[individual] = distribution_value

      for _ in range(size):
          num = random.random()
          for individual, value in distribution.items():
              if num <= value:
                  self.best_individuals.append(individual)
                  break
    
    #krzyżowanie jednopunktowe
    def single_point_crossover(self, parent1, parent2):
        """Krzyżowanie jednopunktowe dla chromosomów binarnych."""
        child1_chromosoms = []
        child2_chromosoms = []
    
    
        # Iterujemy po każdej zmiennej w chromosomie (bo może być ich kilka)
        for p1_chromo, p2_chromo in zip(parent1.chromosom.chromosoms, parent2.chromosom.chromosoms):
            end_of_range = len(p1_chromo)
            k = random.randint(1, end_of_range - 1)  # Punkt krzyżowania (nie może być 0)

    
            # Tworzymy nowe chromosomy dzieci
            new_p1 = p1_chromo[:k] + p2_chromo[k:]
            new_p2 = p2_chromo[:k] + p1_chromo[k:]
    
            child1_chromosoms.append(new_p1)
            child2_chromosoms.append(new_p2)
    
        # Tworzymy nowe osobniki
        child1 = Individual(parent1.chromosom.precision, parent1.variables_count, parent1.chromosom.start_, parent1.chromosom.end_)
        child2 = Individual(parent2.chromosom.precision, parent2.variables_count, parent2.chromosom.start_, parent2.chromosom.end_)
    
        # Podmieniamy chromosomy na nowe
        child1.chromosom.chromosoms = child1_chromosoms
        child2.chromosom.chromosoms = child2_chromosoms
        
        return child1, child2
    
    
    #krzyżowanie dwupunktowe
    def two_point_crossover(self, parent1, parent2, min_gap=1):
        """Krzyżowanie dwupunktowe dla chromosomów binarnych."""
        child1_chromosoms = []
        child2_chromosoms = []
    
        # Iterujemy po każdej zmiennej w chromosomie (bo może być ich kilka)
        for p1_chromo, p2_chromo in zip(parent1.chromosom.chromosoms, parent2.chromosom.chromosoms):
            end_of_range = len(p1_chromo)
            point1 = random.randint(1, end_of_range - 1)  # Pierwszy punkt krzyżowania (nie może być 0)
            while True:
                point2 = random.randint(1, end_of_range - 1)
                if point2 != point1 and abs(point2 - point1) >= min_gap:
                    break
            
            lower = min(point1, point2)
            upper = max(point1, point2)
            # Tworzymy nowe chromosomy dzieci
            new_p1 = p1_chromo[:lower] + p2_chromo[lower:upper] + p1_chromo[upper:]
            new_p2 = p2_chromo[:lower] + p1_chromo[lower:upper] + p2_chromo[upper:]
            
            child1_chromosoms.append(new_p1)
            child2_chromosoms.append(new_p2)
            
        # Tworzymy nowe osobniki
        child1 = Individual(parent1.chromosom.precision, parent1.variables_count, parent1.chromosom.start_, parent1.chromosom.end_)
        child2 = Individual(parent2.chromosom.precision, parent2.variables_count, parent2.chromosom.start_, parent2.chromosom.end_)
    
        # Podmieniamy chromosomy na nowe
        child1.chromosom.chromosoms = child1_chromosoms
        child2.chromosom.chromosoms = child2_chromosoms
        
        return child1, child2
    
    #krzyżowanie jednorodne
    def uniform_crossover(self, parent1, parent2, cross_probability=1.0):
        """Krzyżowanie jednorodne dla chromosomów binarnych."""
        child1_chromosoms = []
        child2_chromosoms = []      
        
        
        for p1_chromo, p2_chromo in zip(parent1.chromosom.chromosoms, parent2.chromosom.chromosoms):
            child1_genes = []
            child2_genes = []
            
            for gene1, gene2 in zip(p1_chromo, p2_chromo):  
                gene_random_rate = random.uniform(0, 1)
    
                if gene_random_rate <= cross_probability:
                    # Zamieniamy geny
                    child1_genes.append(gene2)
                    child2_genes.append(gene1)
                else:
                    # Zostają takie same
                    child1_genes.append(gene1)
                    child2_genes.append(gene2)
            
            # Dodajemy nowo utworzone chromosomy do listy chromosomów dzieci
            child1_chromosoms.append(child1_genes)
            child2_chromosoms.append(child2_genes)
    
        # Tworzymy nowe osobniki
        child1 = Individual(parent1.chromosom.precision, parent1.variables_count, parent1.chromosom.start_, parent1.chromosom.end_)
        child2 = Individual(parent2.chromosom.precision, parent2.variables_count, parent2.chromosom.start_, parent2.chromosom.end_)
    
        # Podmieniamy chromosomy na nowe
        child1.chromosom.chromosoms = child1_chromosoms
        child2.chromosom.chromosoms = child2_chromosoms
    
        return child1, child2 
    
    #krzyżowanie ziarniste
    def discrete_crossover(self, parent1, parent2):
        """Krzyżowanie ziarniste dla chromosomów binarnych."""
        child1prim_chromosoms = []   
        
        
        for p1_chromo, p2_chromo in zip(parent1.chromosom.chromosoms, parent2.chromosom.chromosoms):
            child1prim_genes = []
            
            for gene1, gene2 in zip(p1_chromo, p2_chromo):  
                gene_random_rate = random.uniform(0, 1)
    
                if gene_random_rate <= 0.5:
                    child1prim_genes.append(gene1)
                else:
                    child1prim_genes.append(gene2)
            
            # Dodajemy nowo utworzone chromosomy do listy chromosomów dzieci
            child1prim_chromosoms.append(child1prim_genes)
    
        # Tworzymy nowe osobniki
        child1prim = Individual(parent1.chromosom.precision, parent1.variables_count, parent1.chromosom.start_, parent1.chromosom.end_)

        # Podmieniamy chromosomy na nowe
        child1prim.chromosom.chromosoms = child1prim_chromosoms
        return child1prim
    

    # Real-Valued Crossover Operators
    def arithmetic_crossover_real(self, parent1, parent2, alpha=0.5):
        child1_values = [(1 - alpha) * v1 + alpha * v2 for v1, v2 in zip(parent1.chromosom.real_values, parent2.chromosom.real_values)]
        child2_values = [alpha * v1 + (1 - alpha) * v2 for v1, v2 in zip(parent1.chromosom.real_values, parent2.chromosom.real_values)]
        child1 = Individual(self.precision, self.variables_count, self.start_, self.end_, "real")
        child2 = Individual(self.precision, self.variables_count, self.start_, self.end_, "real")
        child1.chromosom.real_values = child1_values
        child2.chromosom.real_values = child2_values
        return child1, child2

    def linear_crossover_real(self, parent1, parent2):
        p1 = np.array(parent1.chromosom.real_values)
        p2 = np.array(parent2.chromosom.real_values)
        c1_values = 0.5 * p1 + 0.5 * p2
        c2_values = 1.5 * p1 - 0.5 * p2
        c3_values = -0.5 * p1 + 1.5 * p2
        children = []
        for values in [c1_values, c2_values, c3_values]:
            child = Individual(self.precision, self.variables_count, self.start_, self.end_, "real")
            child.chromosom.real_values = np.clip(values, self.start_, self.end_).tolist()
            children.append(child)
        return random.sample(children, 2)

    def blend_crossover_alpha_real(self, parent1, parent2, alpha=0.5):
        child1_values = []
        child2_values = []
        for v1, v2 in zip(parent1.chromosom.real_values, parent2.chromosom.real_values):
            d = abs(v1 - v2)
            lower = min(v1, v2) - alpha * d
            upper = max(v1, v2) + alpha * d
            child1_values.append(random.uniform(lower, upper))
            child2_values.append(random.uniform(lower, upper))
        child1 = Individual(self.precision, self.variables_count, self.start_, self.end_, "real")
        child2 = Individual(self.precision, self.variables_count, self.start_, self.end_, "real")
        child1.chromosom.real_values = np.clip(child1_values, self.start_, self.end_).tolist()
        child2.chromosom.real_values = np.clip(child2_values, self.start_, self.end_).tolist()
        return child1, child2

    def blend_crossover_alpha_beta_real(self, parent1, parent2, alpha=0.5, beta=0.5):
        child1_values = []
        child2_values = []
        for v1, v2 in zip(parent1.chromosom.real_values, parent2.chromosom.real_values):
            d = abs(v1 - v2)
            lower1 = min(v1, v2) - alpha * d
            upper1 = max(v1, v2) + beta * d
            lower2 = min(v1, v2) - beta * d
            upper2 = max(v1, v2) + alpha * d
            child1_values.append(random.uniform(lower1, upper1))
            child2_values.append(random.uniform(lower2, upper2))
        child1 = Individual(self.precision, self.variables_count, self.start_, self.end_, "real")
        child2 = Individual(self.precision, self.variables_count, self.start_, self.end_, "real")
        child1.chromosom.real_values = np.clip(child1_values, self.start_, self.end_).tolist()
        child2.chromosom.real_values = np.clip(child2_values, self.start_, self.end_).tolist()
        return child1, child2

    def average_crossover_real(self, parent1, parent2):
        child1_values = [(v1 + v2) / 2 for v1, v2 in zip(parent1.chromosom.real_values, parent2.chromosom.real_values)]
        child2_values = [(v1 + v2) / 2 for v1, v2 in zip(parent1.chromosom.real_values, parent2.chromosom.real_values)]
        child1 = Individual(self.precision, self.variables_count, self.start_, self.end_, "real")
        child2 = Individual(self.precision, self.variables_count, self.start_, self.end_, "real")
        child1.chromosom.real_values = child1_values
        child2.chromosom.real_values = child2_values
        return child1, child2


    def population_after_crossover(self, crossover_method_number, elite ,crossover_rate=1.0, cross_probability=0.7):
        """Wykonuje krzyżowanie dla całej populacji."""
        new_population = []
        selected_individuals = self.best_individuals[:]
        random.shuffle(selected_individuals)
        needed_population_size = self.population_size - len(self.best_individuals) - elite
        
        if len(selected_individuals) % 2 != 0:
            selected_individuals.append(random.choice(selected_individuals))

        i=0
        while len(new_population)<needed_population_size:
            
            parent1, parent2 = selected_individuals[i % len(selected_individuals)], selected_individuals[(i + 1) % len(selected_individuals)]
    
            if random.random() < crossover_rate:
                if crossover_method_number == 1:
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                elif crossover_method_number == 2:
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                elif crossover_method_number == 3:
                    child1, child2 = self.uniform_crossover(parent1, parent2, cross_probability)
                elif crossover_method_number == 4:
                    child1 = self.discrete_crossover(parent1, parent2)
                elif crossover_method_number == 5:
                    child1, child2 = self.arithmetic_crossover_real(parent1, parent2)
                elif crossover_method_number == 6:
                    child1, child2 = self.linear_crossover_real(parent1, parent2)
                elif crossover_method_number == 7:
                    child1, child2 = self.blend_crossover_alpha_real(parent1, parent2, alpha=0.5)
                elif crossover_method_number == 8:
                    child1, child2 = self.blend_crossover_alpha_beta_real(parent1, parent2, alpha=0.3, beta=0.7)
                elif crossover_method_number == 9:
                    child1, child2 = self.average_crossover_real(parent1, parent2)
                if len(new_population)+1<needed_population_size:
                    try:
                        new_population.extend([child1, child2])  
                    except Exception as e:
                        new_population.extend([child1]) 
                else:
                    new_population.extend([child1]) 
            else:
                new_population.extend([parent1, parent2])
            
            i+=2
                
        self.best_individuals.extend(new_population)
        
    def __str__(self):
        return "\n".join(str(individual) for individual in self.individuals)
    
    def mutate_boundary(self, individual):
        """Mutacja brzegowa – zmienia pierwszy i/lub ostatni bit z określonym prawdopodobieństwem."""
        for c in individual.chromosom.chromosoms:
            if random.random() < 0.5:
                c[0] = 0 if c[0] == 1 else 1
            else:
                c[-1] = 0 if c[-1] == 1 else 1

    def mutate_one_point(self, individual):
        """Mutacja jednopunktowa – losowo zmienia jeden bit."""
        for c in individual.chromosom.chromosoms:
            index = random.randint(0, len(c) - 1)
            
            c[index] = 0 if c[index] == 1 else 1

    def mutate_two_point(self, individual):
        """Mutacja dwupunktowa – losowo zmienia dwa bity."""
        for c in individual.chromosom.chromosoms:
            indexes = random.sample(range(len(c)), 2)
            for idx in indexes:
                c[idx] = 0 if c[idx] == 1 else 1

    def mutate_uniform(self, individual):
        """Mutacja równomierna - losowo zmienia każdy bit z określonym prawdopodobieństwem."""
        for c in individual.chromosom.chromosoms:
            for i in range(len(c)):
                    c[i] = 0 if c[i] == 1 else 1

    def mutate_gaussian(self, individual, std_dev=0.1):
        """Mutacja Gaussa - dodaje losową wartość z rozkładu normalnego do wartości rzeczywistych."""
        if self.representation_type == "real":
            for i in range(len(individual.chromosom.real_values)):
                mutation_value = np.random.normal(0, std_dev)
                individual.chromosom.real_values[i] = np.clip(
                    individual.chromosom.real_values[i] + mutation_value,
                    self.start_[i] if isinstance(self.start_, list) else self.start_,
                    self.end_[i] if isinstance(self.end_, list) else self.end_
                )
        elif self.representation_type == "binary":
            # Mutacja Gaussa nie jest typowa dla reprezentacji binarnej.
            # Można rozważyć konwersję na wartości rzeczywiste, mutację i ponowną konwersję,
            # ale to wykracza poza prostą implementację.
            print("Ostrzeżenie: Mutacja Gaussa nie jest bezpośrednio stosowana do reprezentacji binarnej.")
            


    def population_after_mutationr(self, mutation_method, mutation_rate=1.0):
        for i in range(0, len(self.best_individuals)):

            if random.random() < mutation_rate:
                if mutation_method == "One Point":
                    self.mutate_one_point(self.best_individuals[i])
                elif mutation_method == "Two Point":
                    self.mutate_two_point(self.best_individuals[i])
                elif mutation_method == "Boundary":
                   self.mutate_boundary(self.best_individuals[i])
                elif mutation_method == "Uniform":
                    self.mutate_uniform(self.best_individuals[i])
                elif mutation_method == "Gaussian":
                    self.mutate_gaussian(self.best_individuals[i], self.std_dev)

    def inversion(self, inversion_rate):
        """Operator inwersji – losowo odwraca fragment chromosomu z określonym prawdopodobieństwem."""

        for individual in self.best_individuals:

            for c in individual.chromosom.chromosoms:
                if random.random() < inversion_rate:
                    idx1, idx2 = sorted(random.sample(range(len(c)), 2))
                    
                    c[idx1:idx2 + 1] = list(c[idx1:idx2 + 1])[::-1]

    def elitism(self, elite_percent: float = 0.1, elite_count: int = None):

        """Strategia elitarna – wybiera najlepsze osobniki do nowej populacji."""

        sorted_population = sorted(self.individuals, key=lambda ind: self.fitness(ind), reverse=self.optimum) #reverse=self.optimum
        elite_num = elite_count if elite_count else int(self.population_size * elite_percent)


        elite_num = max(1, elite_num)


        return sorted_population[:elite_num]  