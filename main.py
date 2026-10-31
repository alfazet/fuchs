from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List
from enum import Enum

def is_power_of_two(n):
    return n > 0 and n.bit_count() == 1


class DecodeResult(Enum):
    SUCCESS = 0
    REPAIRED = 1
    ERROR = 2

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
        n = 2**r - 1 # codeword length
        k = n - r #  message length
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

        syndromes = {}
        zero_syndrome = tuple(np.zeros(r, dtype=int))
        syndromes[zero_syndrome] = np.zeros(n, dtype=int)
        for i in range(n):
            e = np.zeros(n, dtype=int)
            e[i] = 1
            s = tuple(self.syndrome(e))
            syndromes[s] = e

        self.syndromes = syndromes

    def syndrome(self, message: np.ndarray) -> np.ndarray:
        return (self.H @ message) % 2

    def encode(self, data: np.ndarray) -> np.ndarray:
        return (data @ self.G) % 2
    
    def decode_concat(self, data: np.ndarray) -> np.ndarray:
        pass

    def decode(self, received: np.ndarray) -> tuple[np.ndarray, DecodeResult]:
        if len(received) != self.n:
            raise SizeError("Decode attempt of message with invalid length")

        error = self.syndromes.get(tuple(self.syndrome(received)), None)
        print(error)
        if error is None:
            return np.array([], dtype=int), DecodeResult.ERROR
        else:
            if np.any(error):
                corrected = (received + error) % 2
                return [corrected[:self.k], DecodeResult.REPAIRED]
            else:
                return [received[:self.k], DecodeResult.SUCCESS]

def run_example(codeword):
    code = HammingCode(3)
    print("G matrix")
    print(code.G)
    print("H matrix")
    print(code.H)
    print("codeword: ", codeword)
    message = code.encode(codeword)
    print("message: ", message)
    print("syndrome: ", code.syndrome(message))
    print("decode: ", code.decode(message))

    message[0] = 0 if message[0] else 1
    print("message after flip: ", message)
    print("syndrome after flip: ", code.syndrome(message))
    print("decode: ", code.decode(message))

    message[1] = 0 if message[1] else 1
    print("message after flip: ", message)
    print("syndrome after flip: ", code.syndrome(message))
    print("decode: ", code.decode(message))

def main():
    run_example([1, 0, 1, 1])

if __name__ == "__main__":
    main()


