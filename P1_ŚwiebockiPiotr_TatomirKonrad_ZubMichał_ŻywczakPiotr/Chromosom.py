from math import log2, ceil
import random

class Chromosom:
    def __init__(self, precision, variables_count, start_, end_):
        self.precision = precision
        self.variables_count = variables_count
        self.start_ = start_
        self.end_ = end_
        self.chromosom_len = ceil(self.precision * log2(self.end_ - self.start_))
        self.chromosoms = self._generate_chromosom() 

    def _generate_chromosom(self) -> list:
      chromosoms = []
      for i in range(self.variables_count):
        chromosom = []
        for i in range(self.chromosom_len):
            chromosom.append(random.randint(0, 1))
        chromosoms.append(chromosom)
      return chromosoms

    def _decode_chromosom(self) -> list:
        decoded_chromosom = []
        for chromosom in self.chromosoms:
            decimal_number = sum(bit * (2 ** i) for i, bit in enumerate(reversed(chromosom)))
            decoded = self.start_ + decimal_number * (self.end_ - self.start_) / (2 ** self.chromosom_len - 1)
            decoded_chromosom.append(float(decoded))
        return decoded_chromosom

    def __str__(self):
        return f"Chromosoms: {self.chromosoms} | Value in Decimal: {self.decoded_chromosom}"
