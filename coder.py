import math
import numpy as np

import hamming as hm
import image_io as iio

def _str2bits(s: str) -> np.ndarray:
    enc = bytearray()
    enc.extend(map(ord, s))
    bits = hm.as_bits(enc)
    return bits

def _bits2str(bits: np.ndarray) -> str:
    arr = np.packbits(bits)
    s = arr.tobytes().decode("utf-8")
    return s

# Hamming code encoding function; inactive for now
def _encode(bits: np.ndarray) -> np.ndarray:
    return bits

# Hamming code decoding function; inactive for now
def _decode(bits: np.ndarray) -> np.ndarray:
    return bits

def _rectangulate(bits: np.ndarray, w: int, l: int = -1) -> np.ndarray:
    # Compute length if not given (or invalid)
    if l < 1:
       l = int(math.ceil(len(bits) / w))

    if len(bits) > w * l:
        raise ValueError('Too many bits to be encoded in rectangle with given dimensions.')

    if len(bits) < w * l:
        bits = np.concatenate([bits, np.zeros(w * l - len(bits))]).astype(np.uint8)

    rect = np.reshape(bits, (l, w))
    return rect

def encode_to_image(s: str, path, w: int, l: int = -1) -> None:
    bits = _str2bits(s)
    enc_bits = _encode(bits)
    rect = _rectangulate(enc_bits, w, l)
    iio.write_bits(rect, path)

def decode_from_image(path) -> str:
    rect = iio.read_bits(path)
    return 'asdf'
