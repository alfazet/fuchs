import numpy as np
from hamming import HammingCode, pad_to_multiple
from collections import Counter


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


def run_example2(payload=b"MAGAZYN-12"):
    code = HammingCode(3)
    # print("G matrix")
    # print(code.G)
    # print("H matrix")
    # print(code.H)

    data = pad_to_multiple(payload, code.k)
    coded = code.encode(data)

    def show(label, bits):
        out, res = code.decode(bits)
        text = bytes(np.packbits(out[: 8 * len(payload)]))
        print(f"{label:24} {text!r:16} {dict(Counter(r.name for r in res))}")

    show("clean", coded)
    coded[0] ^= 1
    show("flip cell 0", coded)
    coded[17] ^= 1
    show("+ flip cell 17", coded)
    coded[18] ^= 1
    show("+ flip cell 18", coded)


def main():
    print("--- example 1 ---")
    run_example([1, 0, 1, 1])
    print("\n--- example 2 ---")
    run_example2()


if __name__ == "__main__":
    main()
