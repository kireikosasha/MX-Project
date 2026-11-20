"""Museum guard's alert level.

A guard watches one small gallery that holds at most 3 visitors and starts empty,
with the guard "calm". Reading the tokens in order: "enter" adds a visitor unless
the gallery already holds 3 (then the visitor is turned away and nothing changes);
"exit" removes a visitor unless the gallery is empty (then nothing changes);
"flash" is a camera flash, which raises the guard's alert by one step (at most to
the top level) only if at least one visitor is in the gallery at that moment, and
is ignored when the gallery is empty; "patrol" is the guard walking a round, which
lowers the alert by one step (never below the bottom level). The alert levels from
bottom to top are "calm", "watchful", "tense", "alarm", and the answer is the
guard's alert level after the whole sequence has been read.
"""

ALPHABET = ["enter", "exit", "flash", "patrol"]

_LEVELS = ["calm", "watchful", "tense", "alarm"]
_CAPACITY = 3


def answer(seq):
    visitors = 0
    alert = 0
    for tok in seq:
        if tok == "enter":
            if visitors < _CAPACITY:
                visitors += 1
        elif tok == "exit":
            if visitors > 0:
                visitors -= 1
        elif tok == "flash":
            if visitors > 0 and alert < len(_LEVELS) - 1:
                alert += 1
        elif tok == "patrol":
            if alert > 0:
                alert -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _LEVELS[alert]
