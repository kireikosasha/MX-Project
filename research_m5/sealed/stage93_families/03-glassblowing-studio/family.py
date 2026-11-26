"""Glassblowing studio.

A gaffer works one piece at a time on the blowpipe, and the tokens are the
moves in order. "gather" dips the pipe in the furnace and adds one gather of
glass to the piece; on an empty pipe it starts a new piece of one gather that
has just been heated. "reheat" puts the piece in the glory hole so it is fully
hot again. "blow" works the piece and cools it: since its last heating a piece
can take as many blows as it currently has gathers, and the blow after that
cracks it, so the whole piece is lost and the pipe is empty. Gathering does
not reheat the piece; it only makes it thicker, so it can take more blows.
"anneal" puts the piece into the annealer, where all its gathers are saved,
and leaves the pipe empty. "blow", "reheat" and "anneal" do nothing on an
empty pipe, and a piece still on the pipe at the end is not saved. The
answer is the total number of gathers saved in the annealer, as a decimal
string.
"""

ALPHABET = ["gather", "blow", "reheat", "anneal"]


def answer(seq):
    gathers = 0          # gathers on the current piece (0 = empty pipe)
    blows = 0            # blows since the current piece was last heated
    saved = 0            # gathers saved in the annealer
    for tok in seq:
        if tok == "gather":
            if gathers == 0:
                blows = 0
            gathers += 1
        elif gathers == 0:
            continue
        elif tok == "blow":
            blows += 1
            if blows > gathers:
                gathers = 0
                blows = 0
        elif tok == "reheat":
            blows = 0
        elif tok == "anneal":
            saved += gathers
            gathers = 0
            blows = 0
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(saved)
