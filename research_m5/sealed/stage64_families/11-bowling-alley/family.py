"""Whose turn is it on lane 7?

Three friends, Ann, Bo and Cy, share one bowling lane. Play starts with Ann
about to throw the first ball of her frame, and the order of play is
Ann -> Bo -> Cy -> Ann -> ... Each token is what happened on the lane next.
"strike": the current bowler's frame ends at once and the next bowler in the
current order steps up for a first ball. "pins": an ordinary ball; if it was
the bowler's first ball they stay up for a second ball, and if it was their
second ball their frame ends and the next bowler in order steps up. "foul":
the line judge reverses the order of play (so Ann -> Cy -> Bo -> Ann if it was
forward, and back again on the next foul), and then the ball counts exactly
like "pins" under the new order. "jam": the pinsetter jams and re-racks; the
same bowler starts their frame over from the first ball, and the order is
unchanged. The answer is the name of the bowler who throws the next ball
after the whole sequence: "Ann", "Bo" or "Cy".
"""

ALPHABET = ["strike", "pins", "foul", "jam"]

_NAMES = ["Ann", "Bo", "Cy"]


def answer(seq):
    bowler = 0      # index into _NAMES
    ball = 1        # 1 or 2
    step = 1        # +1 forward order, -1 reversed order
    for tok in seq:
        if tok == "strike":
            bowler = (bowler + step) % 3
            ball = 1
        elif tok == "pins" or tok == "foul":
            if tok == "foul":
                step = -step
            if ball == 1:
                ball = 2
            else:
                bowler = (bowler + step) % 3
                ball = 1
        elif tok == "jam":
            ball = 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _NAMES[bowler]
