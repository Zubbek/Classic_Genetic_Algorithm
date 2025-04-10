from Chromosom import Chromosom

class Individual:
    def __init__(self, precision, variables_count, start_, end_, representation_type="binary"):
        self.chromosom = Chromosom(precision, variables_count, start_, end_, representation_type)
        self.variables_count = variables_count

    def __str__(self):
        return f"{self.chromosom}"