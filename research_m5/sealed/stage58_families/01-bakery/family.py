"""Bakery counter: how many customers leave empty-handed.

The day starts with no dough on the bench and no loaves on the shelf. Read the
tokens in order. "mix" puts one more piece of dough on the bench. "bake" puts
every piece of dough on the bench into the oven and moves the resulting loaves
to the shelf; the shelf holds at most 5 loaves, so any loaves beyond 5 are
given away and lost, and the bench is left empty. "buy" is a customer: if the
shelf has a loaf they take one, otherwise they leave empty-handed. "night"
throws away every loaf on the shelf (they went stale) but leaves the dough on
the bench untouched. The answer is the number of customers who left
empty-handed, written as a decimal string (e.g. "0", "3").
"""

ALPHABET = ["mix", "bake", "buy", "night"]

SHELF_CAPACITY = 5


def answer(seq):
    bench = 0
    shelf = 0
    turned_away = 0
    for tok in seq:
        if tok == "mix":
            bench += 1
        elif tok == "bake":
            shelf = min(SHELF_CAPACITY, shelf + bench)
            bench = 0
        elif tok == "buy":
            if shelf > 0:
                shelf -= 1
            else:
                turned_away += 1
        elif tok == "night":
            shelf = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(turned_away)
