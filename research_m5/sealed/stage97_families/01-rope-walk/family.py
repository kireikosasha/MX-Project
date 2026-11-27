"""Rope walk tally.

A spinner works down a long rope walk while a boy turns the wheel. The wheel
starts turning for an S twist and no yarn is on the go. Each "spin" means the
spinner walks back one pace, so the yarn being spun grows by one pace. "turn"
means the boy reverses the wheel, switching the twist between S and Z (the yarn
on the go is not affected yet). "hook" means the yarn on the go is taken off and
hung on the rack, and all of its paces are credited to the twist the wheel has
at that moment; the spinner then starts a new, empty yarn (hooking an empty yarn
credits nothing). "snap" means the yarn on the go breaks and is thrown away, so
its paces are lost and a new, empty yarn begins; the wheel is not affected. Yarn
still on the go at the end of the sequence is not counted. The answer is the
twist with more paces hung on the rack in total: "S", "Z", or "even" if the two
totals are equal (including when nothing was hung).
"""

ALPHABET = ["spin", "turn", "hook", "snap"]


def answer(seq):
    twist = "S"
    current = 0
    totals = {"S": 0, "Z": 0}
    for tok in seq:
        if tok == "spin":
            current += 1
        elif tok == "turn":
            twist = "Z" if twist == "S" else "S"
        elif tok == "hook":
            totals[twist] += current
            current = 0
        elif tok == "snap":
            current = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if totals["S"] > totals["Z"]:
        return "S"
    if totals["Z"] > totals["S"]:
        return "Z"
    return "even"
