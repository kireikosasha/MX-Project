"""Ferris-wheel rider count.

A small ferris wheel has 4 gondolas arranged in a circle, all empty at the
start, with gondola 0 at the loading platform at the bottom. The sequence is
read left to right. "turn" rotates the wheel one step forward, so the next
gondola (0 -> 1 -> 2 -> 3 -> 0 ...) arrives at the platform. "board" puts a
rider into the gondola currently at the platform if it is empty (an already
occupied gondola is left unchanged). "exit" lets the rider out of the gondola
at the platform if it is occupied (an empty gondola is left unchanged). The
answer is the number of occupied gondolas after the whole sequence, written as
a digit string from "0" to "4".
"""

ALPHABET = ["turn", "board", "exit"]

N_GONDOLAS = 4


def answer(seq):
    occupied = [False] * N_GONDOLAS
    pos = 0
    for tok in seq:
        if tok == "turn":
            pos = (pos + 1) % N_GONDOLAS
        elif tok == "board":
            occupied[pos] = True
        elif tok == "exit":
            occupied[pos] = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(sum(occupied))
