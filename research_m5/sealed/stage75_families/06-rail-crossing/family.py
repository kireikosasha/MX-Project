"""Railway level crossing: barrier and waiting cars.

Read the tokens in order at a level crossing that starts with the barrier up,
no train in the section and no car waiting.  "train" means one more train has
entered the protected section, so the barrier is (or stays) down.  "clear"
means one train has left the section; if there was no train in the section it
changes nothing, and when the last train leaves the barrier rises and every
waiting car crosses, so the queue becomes empty.  "car" means a car arrives:
if the barrier is up it simply crosses, otherwise it joins the queue.  The
answer is "open" if the barrier is up at the end, and otherwise "closed:N"
where N is the number of cars waiting at the barrier at the end.
"""

ALPHABET = ("train", "clear", "car")


def answer(seq):
    trains = 0
    waiting = 0
    for tok in seq:
        if tok == "train":
            trains += 1
        elif tok == "clear":
            if trains > 0:
                trains -= 1
                if trains == 0:
                    waiting = 0
        elif tok == "car":
            if trains > 0:
                waiting += 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if trains == 0:
        return "open"
    return "closed:%d" % waiting
