"""Canal lock with a small upper pound.

A single lock chamber sits between an upper pound (a small reservoir) and the
river below. The chamber starts LOW and the upper pound starts holding 2 units
of water (it can hold at most 3). Tokens are read in order. "rain" adds one unit
to the upper pound (never above 3). "up" is a boat that wants to climb: it
needs one unit from the upper pound to fill the chamber behind it; if the pound
is empty the boat is turned away and nothing changes, otherwise the pound loses
one unit and the chamber ends HIGH (draining a high chamber to let it in costs
nothing). "down" is a boat that wants to descend: if the chamber is already
HIGH it simply enters and the chamber is drained to LOW at no cost; if the
chamber is LOW it must first be filled, which takes one unit from the pound
(the boat is turned away and nothing changes if the pound is empty), and then
it is drained to LOW. The answer is the final state written as
"<chamber>-<pound>", e.g. "high-2" or "low-0".
"""

ALPHABET = ["up", "down", "rain"]

_CAP = 3


def answer(seq):
    chamber = "low"
    pound = 2
    for tok in seq:
        if tok == "rain":
            if pound < _CAP:
                pound += 1
        elif tok == "up":
            if pound > 0:
                pound -= 1
                chamber = "high"
        elif tok == "down":
            if chamber == "high":
                chamber = "low"
            elif pound > 0:
                pound -= 1
                chamber = "low"
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%s-%d" % (chamber, pound)


if __name__ == "__main__":
    import random
    import time

    tests = [
        (["rain"], "low-3"),
        (["up"], "high-1"),
        (["down"], "low-1"),
        (["up", "down"], "low-1"),
        (["down", "down", "down"], "low-0"),
        (["up", "up", "up", "rain", "up"], "high-0"),
        (["rain", "rain", "rain", "up", "down", "up"], "high-1"),
        (["down", "down", "up", "down"], "low-0"),
    ]
    for s, exp in tests:
        got = answer(s)
        assert got == exp, (s, got, exp)
        print(s, "->", got)

    rng = random.Random(0)
    vals = {}
    for _ in range(200):
        s = [rng.choice(ALPHABET) for _ in range(20)]
        a = answer(s)
        vals[a] = vals.get(a, 0) + 1
    print("distinct answers over 200 random len-20:", len(vals), sorted(vals.items()))
    assert len(vals) >= 2

    s = [rng.choice(ALPHABET) for _ in range(1000)]
    t0 = time.perf_counter()
    a = answer(s)
    dt = time.perf_counter() - t0
    print("len 1000 ->", a, "in %.6f s" % dt)
    assert dt < 0.1
    print("all ok")
