"""Ballet school: which wall does the dancer face at the end of the combination?

A student starts in the centre of the studio, standing flat and facing the
mirror. The walls, going clockwise, are mirror, window, barre, piano. Each
token is one step called by the teacher: "releve" rises onto demi-pointe
(staying up if already up); "plie" bends the knees and brings the dancer down
flat; "turn" is a quarter turn clockwise, but only if the dancer is up on
demi-pointe -- from flat feet she cannot spin and nothing happens; "jump"
always lands flat, and if the dancer took off from flat (knees prepared) she
makes a half turn in the air, whereas from demi-pointe she simply drops down
without turning. The answer is the wall the dancer faces after the last step:
"mirror", "window", "barre" or "piano".
"""

ALPHABET = ["plie", "releve", "turn", "jump"]

_WALLS = ["mirror", "window", "barre", "piano"]


def answer(seq):
    facing = 0      # index into _WALLS, clockwise
    up = False      # on demi-pointe?
    for tok in seq:
        if tok == "releve":
            up = True
        elif tok == "plie":
            up = False
        elif tok == "turn":
            if up:
                facing = (facing + 1) % 4
        elif tok == "jump":
            if not up:
                facing = (facing + 2) % 4
            up = False
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _WALLS[facing]
