import itertools
import numpy as np
import pytest

from hamming import CodeError, DecodeResult, HammingCode, as_bits, pad_to_multiple


def all_messages(k):
    return np.array(list(itertools.product([0, 1], repeat=k)), dtype=np.uint8)


@pytest.mark.parametrize("r", [2, 3, 4, 5, 8, 12])
def test_shapes_and_orthogonality(r):
    code = HammingCode(r)
    assert code.n == 2**r - 1
    assert code.k == code.n - r
    assert code.G.shape == (code.k, code.n)
    assert code.H.shape == (r, code.n)
    assert np.all((code.H.astype(int) @ code.G.astype(int).T) % 2 == 0)


@pytest.mark.parametrize("r", [2, 3, 4])
def test_minimum_distance_is_three(r):
    code = HammingCode(r)
    words = (all_messages(code.k).astype(int) @ code.G.astype(int)) % 2
    weights = [w.sum() for w in words if w.sum()]
    assert min(weights) == 3


def test_all_messages_roundtrip():
    code = HammingCode(3)
    for m in all_messages(code.k):
        cw = code.encode_block(m)
        decoded, res = code.decode_block(cw)
        assert res is DecodeResult.SUCCESS
        assert np.array_equal(decoded, m)


def test_every_error_is_repaired():
    code = HammingCode(3)
    for m in all_messages(code.k):
        cw = code.encode_block(m)
        for pos in range(code.n):
            damaged = cw.copy()
            damaged[pos] ^= 1
            decoded, res = code.decode_block(damaged)
            assert res is DecodeResult.REPAIRED, (m, pos)
            assert np.array_equal(decoded, m), (m, pos)


def test_double_error_is_miscorrected():
    code = HammingCode(3)
    n_double, n_miscorrected = 0, 0
    for m in all_messages(code.k):
        cw = code.encode_block(m)
        for a, b in itertools.combinations(range(code.n), 2):
            damaged = cw.copy()
            damaged[a] ^= 1
            damaged[b] ^= 1
            decoded, res = code.decode_block(damaged)
            n_double += 1
            assert res is DecodeResult.REPAIRED
            assert not np.array_equal(decoded, m)
            n_miscorrected += 1
    assert n_double == 16 * 21 and n_miscorrected == n_double


def test_multiblock_roundtrip_and_scattered_damage():
    code = HammingCode(3)
    payload = as_bits(b"MAGAZYN-12")
    padded = pad_to_multiple(payload, code.k)
    cw = code.encode(padded)
    assert len(cw) % code.n == 0

    damaged = cw.copy()
    for block in range(len(cw) // code.n):
        damaged[block * code.n + (block % code.n)] ^= 1

    decoded, results = code.decode(damaged)
    assert all(r is DecodeResult.REPAIRED for r in results)
    assert np.array_equal(decoded[: len(payload)], payload)
    assert bytes(np.packbits(decoded[: len(payload)])) == b"MAGAZYN-12"


def test_concentrated_damage_breaks_one_block():
    code = HammingCode(3)
    padded = pad_to_multiple(as_bits(b"MAGAZYN-12"), code.k)
    cw = code.encode(padded)
    damaged = cw.copy()
    damaged[:2] ^= 1
    decoded, results = code.decode(damaged)
    assert results[0] is DecodeResult.REPAIRED
    assert not np.array_equal(decoded[: code.k], padded[: code.k])


def test_padding_helper():
    bits = as_bits([1, 0, 1])
    out = pad_to_multiple(bits, 4)
    assert len(out) == 4 and np.array_equal(out[:3], bits) and out[3] == 0
