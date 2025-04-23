from math import log2, ceil
import random

class Chromosom:
    def __init__(self, precision, variables_count, start_, end_, representation_type="binary"):
        self.precision = precision
        self.variables_count = variables_count
        self.start_ = start_
        self.end_ = end_
        self.representation_type = representation_type
        if self.representation_type == "binary":
            self.chromosom_len = ceil(self.precision * log2(self.end_ - self.start_))
            self.chromosoms = self._generate_binary_chromosom()
        elif self.representation_type == "real":
            self.real_values = self._generate_real_chromosom()

    def _generate_binary_chromosom(self) -> list:
      chromosoms = []
      for i in range(self.variables_count):
        chromosom = []
        for i in range(self.chromosom_len):
            chromosom.append(random.randint(0, 1))
        chromosoms.append(chromosom)
      return chromosoms
    
    def _generate_real_chromosom(self) -> list:
        return [random.uniform(self.start_, self.end_) for _ in range(self.variables_count)]

    def _decode_binary_chromosom(self) -> list:
        decoded_chromosom = []
        for chromosom in self.chromosoms:
            decimal_number = sum(bit * (2 ** i) for i, bit in enumerate(reversed(chromosom)))
            decoded = self.start_ + decimal_number * (self.end_ - self.start_) / (2 ** self.chromosom_len - 1)
            decoded_chromosom.append(float(decoded))
        return decoded_chromosom
    
    def decode(self) -> list:
        if self.representation_type == "binary":
            return self._decode_binary_chromosom()
        elif self.representation_type == "real":
            return self.real_values
        return []

    def __str__(self):
        if self.representation_type == "binary":
            return f"Binary Chromosoms: {self.chromosoms} | Value in Decimal: {self.decode()}"
        elif self.representation_type == "real":
            return f"Real Values: {self.real_values}"
        return "Empty Chromosom"