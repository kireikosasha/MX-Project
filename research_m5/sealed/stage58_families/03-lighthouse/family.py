"""Lighthouse keeper's wreck log.

The sequence is the keeper's night-and-day log, read in order. The lamp starts
dark and the sea starts clear. "dusk" lights the lamp and "dawn" puts it out;
"fog" brings fog in and "clear" lifts it (repeating either changes nothing).
Each "ship" passes the rock: it is wrecked exactly when the fog is in and the
lamp is dark at that moment; otherwise it passes safely. The answer is the
total number of wrecked ships over the whole log, written as a decimal string
(for example "0", "2", "17").
"""

ALPHABET = ["dusk", "dawn", "fog", "clear", "ship"]


def answer(seq):
    lamp_lit = False
    foggy = False
    wrecks = 0
    for tok in seq:
        if tok == "dusk":
            lamp_lit = True
        elif tok == "dawn":
            lamp_lit = False
        elif tok == "fog":
            foggy = True
        elif tok == "clear":
            foggy = False
        elif tok == "ship":
            if foggy and not lamp_lit:
                wrecks += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(wrecks)
