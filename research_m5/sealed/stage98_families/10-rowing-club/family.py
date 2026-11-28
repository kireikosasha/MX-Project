"""Rowing club dock.

The tokens describe, in order, what happens on the landing stage of a rowing
club that starts the session with nobody waiting. "rower" means one rower
arrives and waits on the dock; "cox" means one coxswain arrives and waits.
"pair" means a coxless pair tries to launch: if at least two rowers are
waiting, two of them leave in the boat, otherwise nothing happens. "four"
means a coxed four tries to launch: if at least four rowers and at least one
coxswain are waiting, four rowers and one coxswain leave in the boat,
otherwise nothing happens (nobody leaves). Boats that have launched never
come back. The answer is who is still waiting on the dock at the end,
written as "<R> rowers <C> coxes", for example "3 rowers 1 coxes".
"""

ALPHABET = ["rower", "cox", "pair", "four"]


def answer(seq):
    rowers = 0
    coxes = 0
    for tok in seq:
        if tok == "rower":
            rowers += 1
        elif tok == "cox":
            coxes += 1
        elif tok == "pair":
            if rowers >= 2:
                rowers -= 2
        elif tok == "four":
            if rowers >= 4 and coxes >= 1:
                rowers -= 4
                coxes -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%d rowers %d coxes" % (rowers, coxes)
