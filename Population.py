from Individual import Individual
import random

class Population:
    def __init__(self, variables_count, population_size, precision, start_, end_, func, optimum):
        self.variables_count = variables_count
        self.population_size = population_size
        self.func = func 
        self.individuals = [Individual(precision, variables_count, start_, end_) for _ in range(self.population_size)]
        self.optimum = 0 if optimum == "min" else 1
        self.cell = self.getCell()
        self.precision = precision

    def getCell(self) -> dict:
        """Oblicza wartości funkcji celu dla populacji."""
        cell_dict = {}
        X = self.getX()
        for x in X:
            x_tuple = tuple(x)  
            cell_value = self.func(x) 
            cell_dict[x_tuple] = cell_value
            

        return cell_dict
        
    def getSortedCell(self) -> list[tuple[tuple[float], float]]:
        """Sortuje wartości funkcji celu."""
        self.cell=self.getCell() 
        return sorted(self.cell.items(), key=lambda item: item[1], reverse=self.optimum)

    def getBestBySelection(self, percentage: float,) -> list:
        """Zwraca najlepsze osobniki według funkcji celu."""
        size = int(percentage * self.population_size / 100)
        # size = int(percentage * self.population_size)
        
        best_individuals = self.getSortedCell()
        return best_individuals[:size]
    
    def getBestByTournament(self, k) -> list:
      """Zwraca najlepsze osobniki według funkcji celu."""
      if k > self.population_size:
          raise ValueError("k parameter cannot be greater than population_size!")
      
      cell = self.cell.copy()
      best_individuals = []

      for _ in range(k):
          tournament = random.sample(list(cell.items()), int(self.population_size / k))  # Losujemy podzbiór
          if self.optimum == 1:
            best = max(tournament, key=lambda x: x[1])  # Wybieramy najlepszego osobnika (dla maksymalizacji)
          else:
            best = min(tournament, key=lambda x: x[1])  # Wybieramy najlepszego osobnika (dla minimalizacji)
          best_individuals.append(best)
          del cell[best[0]]  # Usuwamy wybranego osobnika

      return best_individuals

    def getBestByRulet(self) -> list:
        """Zwraca najlepsze osobniki według funkcji celu, unikając ujemnych wartości."""
        cell = self.cell.copy()
        best_individuals = []

        # Jeśli minimalizujemy, zamieniamy wartości na odwrotności
        if self.optimum == 0:
            cell = {key: 1 / value for key, value in cell.items()}

        # Znajdujemy najmniejszą wartość funkcji celu
        min_value = min(cell.values())

        # Jeśli wartości są ujemne, dodajemy stałą przesunięcia
        if min_value < 0:
            shift_constant = abs(min_value) + 1  # Zapewniamy, że wszystkie wartości będą dodatnie
            cell = {key: value + shift_constant for key, value in cell.items()}

        total_fitness = sum(cell.values())
        if total_fitness == 0:
            raise ValueError("Total fitness is zero!")

        probabilities = {}
        distribution = {}
        distribution_value = 0

        for value in cell.values():
            probability = value / total_fitness
            probabilities[value] = probability
            distribution_value += probability
            distribution[value] = distribution_value

        # Poprawiona walidacja sumy prawdopodobieństw
        if not (0.9999 <= sum(probabilities.values()) <= 1.0001):
            raise ValueError("Sum of probabilities is not approximately 1!")

        last_item = list(distribution.items())[-1][1]
        if not (0.9999 <= last_item <= 1.0001):
            raise ValueError("Last distribution value is not approximately 1!")

        num = random.random()
        for key, value in distribution.items():
            if num <= value:
                # Znalezienie pierwszego osobnika o tej wartości
                candidates = [k for k, v in cell.items() if v == key]
                x = random.choice(candidates)  # Jeśli kilku, losujemy jednego
                best_individuals.append((x, self.cell[x]))  # Zachowujemy w oryginalnej skali
                break  # Wybieramy jednego, nie usuwamy z `cell`

        return best_individuals
    
    #krzyżowanie jednopunktowe
    def single_point_crossover(self, parent1, parent2):
        """Krzyżowanie jednopunktowe dla chromosomów binarnych."""
        child1_chromosoms = []
        child2_chromosoms = []
    
        # print("Parent 1 chromosoms:", parent1.chromosom.chromosoms)
        # print("Parent 2 chromosoms:", parent2.chromosom.chromosoms)
    
        # Iterujemy po każdej zmiennej w chromosomie (bo może być ich kilka)
        for p1_chromo, p2_chromo in zip(parent1.chromosom.chromosoms, parent2.chromosom.chromosoms):
            end_of_range = len(p1_chromo)
            k = random.randint(1, end_of_range - 1)  # Punkt krzyżowania (nie może być 0)
            # print(f"Crossover point: {k}")
    
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
    
        # print("Child 1 chromosoms:", child1.chromosom.chromosoms)
        # print("Child 2 chromosoms:", child2.chromosom.chromosoms)
    
        return child1, child2
    
    #krzyżowanie dwupunktowe
    def two_point_crossover(self, parent1, parent2, min_gap=1):
        """Krzyżowanie dwupunktowe dla chromosomów binarnych."""
        child1_chromosoms = []
        child2_chromosoms = []
    
        # print("Parent 1 chromosoms:", parent1.chromosom.chromosoms)
        # print("Parent 2 chromosoms:", parent2.chromosom.chromosoms)
    
        # Iterujemy po każdej zmiennej w chromosomie (bo może być ich kilka)
        for p1_chromo, p2_chromo in zip(parent1.chromosom.chromosoms, parent2.chromosom.chromosoms):
            end_of_range = len(p1_chromo)
            point1 = random.randint(1, end_of_range - 1)  # Pierwszy punkt krzyżowania (nie może być 0)
            while True:
                point2 = random.randint(1, end_of_range - 1)
                if point2 != point1 and abs(point2 - point1) >= min_gap:
                    break
            # print(f"Crossover points: {point1} {point2}")
            
            lower = min(point1, point2)
            upper = max(point1, point2)
            # print(lower, upper)
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
    
        # print("Child 1 chromosoms:", child1.chromosom.chromosoms)
        # print("Child 2 chromosoms:", child2.chromosom.chromosoms)
        
        return child1, child2
    
    def population_after_single_point_crossover(self, crossover_method_number, crossover_rate=1.0):
        """Wykonuje jednopunktowe krzyżowanie dla całej populacji."""
        new_population = []
        individuals = self.individuals[:]
        random.shuffle(individuals)
        
        for i in range(0, len(individuals) - 1, 2):
            parent1, parent2 = individuals[i], individuals[i + 1]
            
            if random.random() < crossover_rate:
                if crossover_method_number == 1:
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                elif crossover_method_number == 2:
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])
        
        self.individuals = new_population
        
    def getX(self) -> list:
        """Zwraca listę fenotypów (wartości zmiennych)."""
        return [[float(x) for x in individual.chromosom._decode_chromosom()] for individual in self.individuals]

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
        new_population = []
        individuals = self.individuals[:]
        # random.shuffle(individuals)
        
        for i in range(0, len(individuals)):

            if random.random() < mutation_rate:
                if mutation_method == "One Point":
                    self.mutate_one_point(individuals[i])
                elif mutation_method == "Two Point":
                    self.mutate_two_point(individuals[i])
                elif mutation_method == "Boundary":
                   self.mutate_boundary(individuals[i])

        # print(self.individuals[0].chromosom.chromosoms)
        self.individuals
        
    def inversion(self, inversion_rate):
        """Operator inwersji – losowo odwraca fragment chromosomu z określonym prawdopodobieństwem."""

        for individual in self.individuals[:]:

            for c in individual.chromosom.chromosoms:
                if random.random() < inversion_rate:
                    idx1, idx2 = sorted(random.sample(range(len(c)), 2))

                    
                    c[idx1:idx2 + 1] = list(c[idx1:idx2 + 1])[::-1]
        
        # print(self.individuals[0].chromosom.chromosoms)
                    
    # def fitness(self, individual: Individual) :
    #     """Funkcja przystosowania, używa rzeczywistej funkcji celu.""" 
    #     # w sumie nie wiem którą wersję wziąść
        
    #     return self.func(individual.chromosom._decode_chromosom())
    #     return self.func(individual.chromosom.decoded_chromosom)
    #     # return sum(x.count(1) for x in individual.chromosom.chromosoms) 

    def fitness(self, individual: Individual):
        """Funkcja przystosowania, używa rzeczywistej funkcji celu."""  
        decoded = individual.chromosom._decode_chromosom()
        
        # Debugowanie, aby zobaczyć, co zwraca dekodowanie
        print("Decoded Chromosome:", decoded, [type(x) for x in decoded])

        # Sprawdzenie, czy wszystkie wartości są float lub Decimal
        try:
            decoded = [float(x) for x in decoded]
        except ValueError:
            raise ValueError(f"Błąd konwersji: {decoded}")

        return self.func(decoded)

    def elitism(self, elite_percent: float = 0.1, elite_count: int = None):

        """Strategia elitarna – wybiera najlepsze osobniki do nowej populacji."""
        sorted_population = sorted(self.individuals, key=lambda ind: self.fitness(ind), reverse=True)

        elite_num = elite_count if elite_count else int(self.population_size * elite_percent)
        elite_num = max(1, elite_num)

        return sorted_population[:elite_num]   




    
if __name__ == "__main__": 
    population = Population(2, 5, 5, -2, 2, lambda x: sum(xi ** 2 for xi in x), "min")

    print(population.individuals[0].chromosom.decoded_chromosom[0])

    print(population.population_after_mutationr("Two Point"),1)

    for i in population.individuals:
        print(i.chromosom.chromosoms)

    print("---------------------------------------------------------")

    
    for ind in population.individuals:
        print(f"Chromosom: {ind.chromosom.chromosoms}, Fitness: {population.fitness(ind)}")
    print("ccc---------------------------------------------------------ccc")

        
    elite =population.elitism()
    print("\nWybrane elitarne osobniki:")
    for ind in elite:
        print(f"Chromosom: {ind.chromosom.chromosoms}, Fitness: {population.fitness(ind)}")