from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List

def is_power_of_two(n):
    return n > 0 and n.bit_count() == 1

class CorrectionCode(ABC):
    """Abstract base class for error-correction codes."""
    @abstractmethod
    def encode(self, data: Union[bytes, List[int]]) -> Union[bytes, List[int]]:
        """Encodes the input data by adding error-correction bits/redundancy."""
        pass

    @abstractmethod
    def decode(self, received: Union[bytes, List[int]]) -> Union[bytes, List[int]]:
        """Decodes the received data, detecting and correcting any correctable errors."""
        pass

# https://en.wikipedia.org/wiki/Hamming_code
class HammingCode(CorrectionCode):
    def __init__(self, r: int):
        if (r < 2): # bound from the other side?
            raise ValueError('Wrong value of r when initaliazing HammingCode.')
        n = 2**r - 1 # message length
        k = n - r # codeword length
        G = np.zeros((k, n), dtype=int)
        H = np.zeros((n - k, n), dtype=int)

        powers_of_two = []
        p = 0
        for i in range(n, 0, -1):
            if is_power_of_two(i):
                powers_of_two.append(i)
            else:
                num_bits = i.bit_length()
                for j in range(num_bits):
                    bit = (i >> j) & 1
                    H[n - k - 1 - j][p] = bit
                p += 1

        for power_of_two in powers_of_two:
            num_bits = power_of_two.bit_length()
            for j in range(num_bits):
                bit = (power_of_two >> j) & 1
                H[n - k - 1 - j][p] = bit
            p += 1

        P = H[:, :k]
        I_k = np.eye(k, dtype=int)
        G = np.hstack((I_k, P.T))

        self.r = r
        self.n = n
        self.k = k
        self.G = G
        self.H = H

    def encode(self, data: List[int]) -> List[int]:
        pass

    def decode(self, received: List[int]) -> List[int]:
        pass

def main():
    code = HammingCode(3)
    print("Hello from fuchs!")

if __name__ == "__main__":
    main()


