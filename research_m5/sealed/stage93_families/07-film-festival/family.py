"""Film festival audience award.

The sequence is a log of one festival hall. The token "drama" means a drama
starts screening and "comedy" means a comedy starts screening; whichever film
started most recently is the one on screen. "cheer" gives one point to the
genre of the film currently on screen and "boo" takes one point away from it
(points can go negative). A "cheer" or "boo" heard before any film has started
is ignored. After the whole log, the answer is the genre with the higher total
of points ("drama" or "comedy"), or "tie" if both totals are equal.
"""

ALPHABET = ["drama", "comedy", "cheer", "boo"]


def answer(seq):
    on_screen = None
    score = {"drama": 0, "comedy": 0}
    for tok in seq:
        if tok == "drama" or tok == "comedy":
            on_screen = tok
        elif tok == "cheer":
            if on_screen is not None:
                score[on_screen] += 1
        elif tok == "boo":
            if on_screen is not None:
                score[on_screen] -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if score["drama"] > score["comedy"]:
        return "drama"
    if score["comedy"] > score["drama"]:
        return "comedy"
    return "tie"
