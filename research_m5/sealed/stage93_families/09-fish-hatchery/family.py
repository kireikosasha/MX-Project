"""Fish hatchery season.

The hatchery starts with an empty incubator, an empty fry pond, and no fish
released or lost. Read the tokens in order: "egg" puts one egg in the
incubator; "hatch" turns one egg from the incubator into one fry in the pond
(nothing happens if the incubator is empty); "net" releases one fry from the
pond into the river (nothing happens if the pond is empty); "heron" lets a
heron eat one fry from the pond (nothing happens if the pond is empty). At the
end, compare the number of fry released into the river with the number eaten
by herons: the answer is "river" if more were released, "heron" if more were
eaten, and "tie" if the two numbers are equal.
"""

ALPHABET = ["egg", "hatch", "net", "heron"]


def answer(seq):
    eggs = 0
    fry = 0
    released = 0
    eaten = 0
    for tok in seq:
        if tok == "egg":
            eggs += 1
        elif tok == "hatch":
            if eggs > 0:
                eggs -= 1
                fry += 1
        elif tok == "net":
            if fry > 0:
                fry -= 1
                released += 1
        elif tok == "heron":
            if fry > 0:
                fry -= 1
                eaten += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if released > eaten:
        return "river"
    if eaten > released:
        return "heron"
    return "tie"
