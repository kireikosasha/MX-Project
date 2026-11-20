"""Chess club ladder.

A club member's evening results are read in order, one token per round:
"win", "loss", "draw" or "bye". The member sits on a club ladder with rungs
1 (bottom) to 5 (top) and starts on rung 3, with a win streak of 0. A "win"
adds one to the win streak; when the streak reaches 2 the member climbs one
rung (never above 5) and the streak goes back to 0. A "loss" drops the member
one rung (never below 1) and resets the streak to 0. A "draw" resets the
streak to 0 without moving. A "bye" changes nothing (the streak is kept).
The answer is the final rung, written as "rung1" .. "rung5".
"""

ALPHABET = ["win", "loss", "draw", "bye"]


def answer(seq):
    rung = 3
    streak = 0
    for tok in seq:
        if tok == "win":
            streak += 1
            if streak == 2:
                rung = min(5, rung + 1)
                streak = 0
        elif tok == "loss":
            rung = max(1, rung - 1)
            streak = 0
        elif tok == "draw":
            streak = 0
        elif tok == "bye":
            pass
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "rung%d" % rung
