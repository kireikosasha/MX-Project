"""Escape room: keys, locks and a flickering light.

A team works through an escape room whose lights keep going out. Reading the
tokens in order: "key" means the team finds one more key and adds it to their
pocket (keys pile up without limit, whether the room is lit or dark); "dark"
means the lights go out and "lamp" means the lights come back on (the room
starts lit, and repeating the current state changes nothing); "lock" means the
team reaches a lock: if the room is lit and they hold at least one key, they
use up one key and that lock is opened, otherwise (dark room, or no keys in
pocket) nothing happens and the lock stays shut. The answer is the total number
of locks opened by the end of the sequence, written as a decimal string.
"""

ALPHABET = ["key", "lock", "dark", "lamp"]


def answer(seq):
    keys = 0
    lit = True
    opened = 0
    for tok in seq:
        if tok == "key":
            keys += 1
        elif tok == "dark":
            lit = False
        elif tok == "lamp":
            lit = True
        elif tok == "lock":
            if lit and keys > 0:
                keys -= 1
                opened += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(opened)
