import random
from math import log2, ceil
from decimal import Decimal
from Individual import Individual


class Population:
    def __init__(self, variables_count, population_size, precision, start_, end_, func, optimum):
        self.variables_count = variables_count
        self.population_size = population_size
        self.func = func
        self.individuals = [Individual(precision, variables_count, start_, end_) for _ in range(self.population_size)]
        self.optimum = 0 if optimum == "min" else 1
        self.precision = precision
        self.best_individuals = []



    def getBestBySelection(self, percentage: float):
          """Zwraca najlepsze osobniki jako listę obiektów Individual."""
          size = int(percentage * self.population_size / 100)
          
          # Sortujemy osobniki według wartości funkcji celu
          self.best_individuals = sorted(
              self.individuals,  # Sortujemy całą populację
              key=lambda individual: float(self.func([float(val) for val in individual.chromosom._decode_chromosom()])),
              reverse=self.optimum  # Jeśli maksymalizujemy, używamy reverse=True
          )
          
          # Wybieramy najlepszych
          self.best_individuals = self.best_individuals[:size]


    def getBestByTournament(self, k):
        """Zwraca najlepsze osobniki według turnieju jako listę obiektów Individual."""
        if k > self.population_size:
            raise ValueError("k parameter cannot be greater than population_size!")

        self.best_individuals = []

        for _ in range(k):
            tournament = random.sample(self.individuals, int(self.population_size / k))  # Losujemy podzbiór

            if self.optimum == 1:
                best = max(tournament, key=lambda individual: float(self.func([float(val) for val in individual.chromosom._decode_chromosom()])))
            else:
                best = min(tournament, key=lambda individual: float(self.func([float(val) for val in individual.chromosom._decode_chromosom()])))

            self.best_individuals.append(best)  # Teraz przechowujemy cały obiekt Individual
            

    def getBestByRulet(self, percentage: float):
      """Zwraca najlepsze osobniki według metody ruletki jako listę obiektów Individual."""
      size = int(percentage * self.population_size / 100)  # Obliczamy liczbę osobników do wybrania
      self.best_individuals = []

      cell = {individual: float(self.func([float(val) for val in individual.chromosom._decode_chromosom()])) for individual in self.individuals}

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
            


    def population_after_mutationr(self, mutation_method, mutation_rate=1.0):

        # random.shuffle(individuals)
        
        for i in range(0, len(self.best_individuals)):

            if random.random() < mutation_rate:
                if mutation_method == "One Point":
                    self.mutate_one_point(self.best_individuals[i])
                elif mutation_method == "Two Point":
                    self.mutate_two_point(self.best_individuals[i])
                elif mutation_method == "Boundary":
                   self.mutate_boundary(self.best_individuals[i])

        # print(self.individuals[0].chromosom.chromosoms)        
    def inversion(self, inversion_rate):
        """Operator inwersji – losowo odwraca fragment chromosomu z określonym prawdopodobieństwem."""

        for individual in self.best_individuals:

            for c in individual.chromosom.chromosoms:
                if random.random() < inversion_rate:
                    idx1, idx2 = sorted(random.sample(range(len(c)), 2))
                    
                    c[idx1:idx2 + 1] = list(c[idx1:idx2 + 1])[::-1]
        

    def fitness(self, individual: Individual):
        """Funkcja przystosowania, używa rzeczywistej funkcji celu."""  
        decoded = individual.chromosom._decode_chromosom()

        return self.func(decoded)

    def elitism(self, elite_percent: float = 0.1, elite_count: int = None):

        """Strategia elitarna – wybiera najlepsze osobniki do nowej populacji."""

        sorted_population = sorted(self.individuals, key=lambda ind: self.fitness(ind), reverse=self.optimum) #reverse=self.optimum
        elite_num = elite_count if elite_count else int(self.population_size * elite_percent)


        elite_num = max(1, elite_num)


        return sorted_population[:elite_num]   


    
if __name__ == "__main__": 
    population = Population(2, 15, 5, -2, 2, lambda x: sum(xi ** 2 for xi in x), "min")

    func =lambda x: sum(xi ** 2 for xi in x)
    # print(population.cell)
    # print(population.individuals[0].chromosom.decoded_chromosom[0])
    # print(population.getSortedCell3())
    
    # print("-------------------------------------------------------")
    # print(population.getCell())
    
    # print("-------------------------------------------------------")

    # print(population.getSortedCell())
    
    # print("-------------------------------------------------------")


    # print(population.population_after_mutationr("Two Point"),1)
    # print(population.best_individuals) over
    print(len(population.individuals))
    elite =population.elitism(0.25)
    print("elite",len(elite))
    population.getBestBySelection(10.0)
    for i in population.best_individuals:
        
        print(i.chromosom.chromosoms, " ", func(i.chromosom._decode_chromosom()))
        
    print("-------------------------------------------------------")
    print(population.population_after_crossover(4,len(elite)))
    print(len(population.best_individuals))
    
    # population.inversion(1)
    # for i in population.best_individuals:
        
    #     print(i.chromosom.chromosoms, " ", func(i.chromosom._decode_chromosom()))
        
    # print("-------------------------------------------------------")

    # for i in population.getBestBySelection2(5.0):
    #     print(i)
    #     print(i.chromosom.chromosoms, " ", func(i.chromosom._decode_chromosom()))
    # # population.elitism(0.7)
    # print("---------------------------------------------------------")
    # population.getBestByRulet(5.0)
    # for i in population.best_individuals:
    #     print(i.chromosom.chromosoms, " ", func(i.chromosom._decode_chromosom()))
    # # for ind in population.individuals:
    # #     print(f"Chromosom: {ind.chromosom.chromosoms}, Fitness: {population.fitness(ind)}")
    # # print("ccc---------------------------------------------------------ccc")
    # print("---------------------------------------------------------")
    # population.getBestByTournament(10,5)
    # for i in population.best_individuals:
    #     print(i.chromosom.chromosoms, " ", func(i.chromosom._decode_chromosom()))
        
    # print("---------------------------------------------------------")
        
    # elite =population.elitism()
    # print("\nWybrane elitarne osobniki:")
    # for ind in elite:
    #     print(f"Chromosom: {ind.chromosom.chromosoms}, Fitness: {population.fitness(ind)}")
    
    # print("---------------------------------------------------------")
    # population.population_after_mutationr("One Point")
    # population.inversion(1)
    # population.individuals=population.best_individuals+elite
    # for i in population.individuals:
    #     print(i.chromosom.chromosoms, " ", func(i.chromosom._decode_chromosom()))