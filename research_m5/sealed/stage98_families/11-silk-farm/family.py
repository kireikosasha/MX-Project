"""Silk farm cocoon count.

A silk farm keeps a tray of waiting silkworms and a pile of fresh mulberry
leaves, both empty at the start, and reads the day's events in order. "hatch"
puts one more waiting worm on the tray. "leaf" adds one leaf to the pile.
"spin" is a spinning call: if at least one worm is waiting and at least one
leaf is on the pile, one worm eats one leaf and spins a cocoon (one worm and
one leaf are removed and the cocoon count goes up by one); otherwise nothing
happens. "storm" soaks the pile and every leaf on it rots (the pile becomes
empty), but the worms are unharmed. The answer is the total number of cocoons
spun over the whole sequence, written as a decimal string such as "0" or "3".
"""

ALPHABET = ["hatch", "leaf", "spin", "storm"]


def answer(seq):
    worms = 0
    leaves = 0
    cocoons = 0
    for tok in seq:
        if tok == "hatch":
            worms += 1
        elif tok == "leaf":
            leaves += 1
        elif tok == "spin":
            if worms > 0 and leaves > 0:
                worms -= 1
                leaves -= 1
                cocoons += 1
        elif tok == "storm":
            leaves = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(cocoons)
