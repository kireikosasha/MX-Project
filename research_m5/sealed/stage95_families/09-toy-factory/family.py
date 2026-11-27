"""Toy factory paint booth.

Toys roll through a paint booth that starts loaded with red paint. The token
'red' or 'blue' reloads the booth with that colour, replacing whatever it held.
The token 'wash' rinses the booth out so it holds no paint at all; toys that go
through an empty booth stay unpainted until 'red' or 'blue' is loaded again.
The token 'doll' sends a doll through the booth and it gets one coat of the
current colour; the token 'truck' sends a truck through and, being big, it gets
two coats of the current colour. Loading or rinsing uses no coats. The answer is
the colour of which more coats were applied over the whole sequence: 'red' or
'blue', or 'tie' when both colours got the same number of coats (including when
no coats were applied at all).
"""

ALPHABET = ["doll", "truck", "red", "blue", "wash"]


def answer(seq):
    paint = "red"      # what the booth currently holds: "red", "blue" or None
    balance = 0        # red coats minus blue coats
    for tok in seq:
        if tok == "red" or tok == "blue":
            paint = tok
        elif tok == "wash":
            paint = None
        elif tok == "doll" or tok == "truck":
            coats = 1 if tok == "doll" else 2
            if paint == "red":
                balance += coats
            elif paint == "blue":
                balance -= coats
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if balance > 0:
        return "red"
    if balance < 0:
        return "blue"
    return "tie"
