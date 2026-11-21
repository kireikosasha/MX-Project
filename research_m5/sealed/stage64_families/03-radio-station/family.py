"""Radio station genre wheel.

A radio station has a genre wheel with three slots in a fixed circle,
rock -> jazz -> pop -> rock, and the wheel starts on "rock" with its
"forward" direction being rock -> jazz -> pop. Reading the broadcast log
token by token: a "jingle" turns the wheel one slot forward; an "ad" turns
it one slot backward (opposite of forward); a "news" bulletin swaps which
direction counts as forward (so later jingles and ads turn the other way);
a "song" is played in whatever genre the wheel currently points at. The
answer is the genre of the most recently played song ("rock", "jazz" or
"pop"), or "silence" if the log contains no song at all.
"""

ALPHABET = ["song", "jingle", "ad", "news"]

_GENRES = ["rock", "jazz", "pop"]


def answer(seq):
    pos = 0          # index into _GENRES; starts on rock
    step = 1         # +1 means forward is rock -> jazz -> pop
    last = "silence"
    for tok in seq:
        if tok == "jingle":
            pos = (pos + step) % 3
        elif tok == "ad":
            pos = (pos - step) % 3
        elif tok == "news":
            step = -step
        elif tok == "song":
            last = _GENRES[pos]
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return last
