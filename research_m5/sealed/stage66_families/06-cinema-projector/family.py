"""Cinema projector: how many frames reached the screen, counted modulo 3.

The projector starts with its lamp on and the film properly threaded. Each token
is read in order: "lamp" flips the lamp (on becomes off, off becomes on);
"jam" makes the film jam in the gate (a jam while already jammed changes
nothing); "thread" re-threads the film, clearing any jam (harmless if the film
was not jammed); "frame" pulls the next frame through the gate, and that frame
reaches the screen only if the lamp is on and the film is not jammed at that
moment. The answer is the number of frames that reached the screen, taken
modulo 3, written as the string "0", "1" or "2".
"""

ALPHABET = ["frame", "lamp", "jam", "thread"]


def answer(seq):
    lamp_on = True
    jammed = False
    shown = 0
    for tok in seq:
        if tok == "lamp":
            lamp_on = not lamp_on
        elif tok == "jam":
            jammed = True
        elif tok == "thread":
            jammed = False
        elif tok == "frame":
            if lamp_on and not jammed:
                shown = (shown + 1) % 3
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(shown)
