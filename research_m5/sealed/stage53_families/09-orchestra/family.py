"""Orchestra stage ensemble.

The stage starts silent. Four sections can play: violin, flute, horn and drum.
Reading the sequence token by token, a section token (violin, flute, horn,
drum) toggles that section: if it is silent it starts playing, if it is
playing it stops. The token "tutti" makes all four sections play at once, and
the token "cut" (the conductor's cut-off) silences every section. After the
whole sequence, the answer names the size of the ensemble that is still
playing: "silence" for 0 sections, "solo" for 1, "duet" for 2, "trio" for 3
and "quartet" for 4.
"""

ALPHABET = ["violin", "flute", "horn", "drum", "tutti", "cut"]

_SECTIONS = ("violin", "flute", "horn", "drum")
_NAMES = ("silence", "solo", "duet", "trio", "quartet")


def answer(seq):
    playing = set()
    for tok in seq:
        if tok == "tutti":
            playing = set(_SECTIONS)
        elif tok == "cut":
            playing = set()
        elif tok in _SECTIONS:
            if tok in playing:
                playing.discard(tok)
            else:
                playing.add(tok)
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _NAMES[len(playing)]
