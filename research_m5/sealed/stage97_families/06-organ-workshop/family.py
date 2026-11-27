"""Pipe-organ workshop: how many pipes end up in the finished rank.

The workshop starts with an empty bench, an empty rank and the stove out
(the room is cold). Read the tokens in order. "cast" puts one newly cast
pipe on the bench. "stove" lights the stove if it is out, or puts it out if
it is lit. "voice" means the voicer takes one pipe from the bench: if the
bench is empty nothing happens; otherwise the pipe leaves the bench, and if
the stove is lit (warm room) it speaks well and is added to the rank, while
if the stove is out (cold room) it speaks badly and is thrown away. The
answer is the number of pipes in the rank after the last token, written as
a decimal number (for example "0" or "3").
"""

ALPHABET = ["cast", "stove", "voice"]


def answer(seq):
    bench = 0
    rank = 0
    warm = False
    for tok in seq:
        if tok == "cast":
            bench += 1
        elif tok == "stove":
            warm = not warm
        elif tok == "voice":
            if bench > 0:
                bench -= 1
                if warm:
                    rank += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(rank)
