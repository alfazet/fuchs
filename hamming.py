from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List
from enum import Enum


Bits = Union[bytes, bytearray, List[int], np.ndarray]


class CodeError(ValueError):
    pass


def as_bits(data: Bits) -> np.ndarray:
    """Coerce bytes/list/ndarray to a validated 1-D uint8 array of 0/1."""
    if isinstance(data, (bytes, bytearray)):
        bits = np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))
    else:
        bits = np.asarray(data)
    if bits.ndim != 1:
        raise CodeError(f"expected a 1-D sequence of bits, got shape {bits.shape}")
    if bits.size and not np.isin(bits, (0, 1)).all():
        raise CodeError("data must contain only 0 and 1")
    return bits.astype(np.uint8)


def pad_to_multiple(bits: Bits, k: int) -> np.ndarray:
    b = as_bits(bits)
    rem = len(b) % k
    return b if rem == 0 else np.concatenate([b, np.zeros(k - rem, dtype=np.uint8)])


def is_power_of_two(n):
    return n > 0 and n.bit_count() == 1


class DecodeResult(Enum):
    SUCCESS = 0
    REPAIRED = 1
    ERROR = 2


class CorrectionCode(ABC):
    """Abstract base class for error-correction codes."""

    @abstractmethod
    def encode(self, data: Bits) -> np.ndarray:
        """Encodes the input data by adding error-correction bits/redundancy."""
        pass

    @abstractmethod
    def decode(self, received: Bits) -> tuple[np.ndarray, List[DecodeResult]]:
        """Decodes the received data, detecting and correcting any correctable errors."""
        pass


# https://en.wikipedia.org/wiki/Hamming_code
class HammingCode(CorrectionCode):
    """(n, k) = (2^r - 1, 2^r - r - 1), min_d = 3"""

    MIN_R, MAX_R = 2, 12

    def __init__(self, r: int):
        if not (self.MIN_R <= r <= self.MAX_R):
            raise CodeError(f"r must be in [{self.MIN_R}, {self.MAX_R}], got {r}")
        n = 2**r - 1  # codeword length
        k = n - r  # message length
        G = np.zeros((k, n), dtype=np.uint8)  # [I_k | P^T]
        H = np.zeros((n - k, n), dtype=np.uint8)  # [P | I_r]

        # column i of H is (n - i) in binary, MSB on top
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
        I_k = np.eye(k, dtype=np.uint8)
        G = np.hstack((I_k, P.T))

        self.r = r
        self.n = n
        self.k = k
        self.G = G
        self.H = H

        syndromes = {}
        zero_syndrome = tuple(np.zeros(r, dtype=np.uint8))
        syndromes[zero_syndrome] = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            e = np.zeros(n, dtype=np.uint8)
            e[i] = 1
            s = tuple(self.syndrome(e))
            syndromes[s] = e

        self.syndromes = syndromes

    def syndrome(self, message: Bits) -> np.ndarray:
        w = as_bits(message)
        if len(w) != self.n:
            raise CodeError(f"expected a codeword of length {self.n}, got {len(w)}")
        return (self.H.astype(np.int64) @ w.astype(np.int64)) % 2

    def encode_block(self, data: Bits) -> np.ndarray:
        m = as_bits(data)
        if len(m) != self.k:
            raise CodeError(f"expected {self.k} message bits, got {len(m)}")
        return ((m.astype(np.int64) @ self.G.astype(np.int64)) % 2).astype(np.uint8)

    def decode_block(self, received: Bits) -> tuple[np.ndarray, DecodeResult]:
        y = as_bits(received)
        if len(y) != self.n:
            raise CodeError("Decode attempt of message with invalid length")

        # y = c + e
        # s = Hy = Hc + He = He
        error = self.syndromes.get(tuple(self.syndrome(y)), None)
        if error is None:
            return np.array([], dtype=np.uint8), DecodeResult.ERROR
        if np.any(error):
            corrected = (y + error) % 2
            return corrected[: self.k], DecodeResult.REPAIRED
        return y[: self.k], DecodeResult.SUCCESS

    def encode(self, data: Bits) -> np.ndarray:
        bits = as_bits(data)
        if len(bits) % self.k:
            raise CodeError(
                f"{len(bits)} bits is not a multiple of k={self.k}; "
                "use pad_to_multiple(bits, code.k) first"
            )
        blocks = bits.reshape(-1, self.k).astype(np.int64)
        return np.concatenate([self.encode_block(b) for b in blocks])

    def decode(self, received: Bits) -> tuple[np.ndarray, List[DecodeResult]]:
        bits = as_bits(received)
        if len(bits) % self.n:
            raise CodeError(f"{len(bits)} bits is not a multiple of n={self.n}")
        out, results = [], []
        for block in bits.reshape(-1, self.n):
            data, res = self.decode_block(block)
            out.append(data)
            results.append(res)
        return (np.concatenate(out) if out else np.zeros(0, dtype=np.uint8)), results
