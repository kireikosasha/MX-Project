"""Archery range, two-lane set match.

Two archers shoot side by side on an archery range, one on the left lane and
one on the right lane. The token "left" means the left archer hits the gold,
"right" means the right archer hits the gold, and "whistle" means the range
officer blows the whistle and closes the current end (a round of shooting).
When an end is closed, the archer with more gold hits in that end wins it and
takes 2 set points; if both have the same number of hits in that end (including
zero each, e.g. two whistles in a row) each takes 1 set point. Hit counts start
again from zero for the next end. When the sequence stops, the end still in
progress is closed as if the whistle had blown (if nobody shot since the last
whistle, it is an empty tied end and changes nothing). The answer is who leads
in set points over the whole match: "left", "right", or "level" if their set
points are equal.
"""

ALPHABET = ["left", "right", "whistle"]


def answer(seq):
    left_end = 0
    right_end = 0
    left_sets = 0
    right_sets = 0

    def close_end():
        nonlocal left_end, right_end, left_sets, right_sets
        if left_end > right_end:
            left_sets += 2
        elif right_end > left_end:
            right_sets += 2
        else:
            left_sets += 1
            right_sets += 1
        left_end = 0
        right_end = 0

    for tok in seq:
        if tok == "left":
            left_end += 1
        elif tok == "right":
            right_end += 1
        elif tok == "whistle":
            close_end()
        else:
            raise ValueError("unknown token: %r" % (tok,))
    close_end()

    if left_sets > right_sets:
        return "left"
    if right_sets > left_sets:
        return "right"
    return "level"
