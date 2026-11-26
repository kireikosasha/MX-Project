"""Cargo ship listing after loading and unloading.

A cargo ship starts empty, with the dock crane working over its port side.
Read the tokens in order: "port" moves the crane over the port side and
"stbd" moves it over the starboard side (moving to the side it is already on
changes nothing); "load" sets one container down on the side the crane is
currently over; "unload" lifts one container off that side, but if that side
holds no containers the crane comes up empty and nothing changes (it never
takes from the other side). After the last token, the answer is the side the
ship lists toward, which is the side holding more containers: "port" or
"stbd", or "even" if both sides hold the same number.
"""

ALPHABET = ["load", "unload", "port", "stbd"]


def answer(seq):
    counts = {"port": 0, "stbd": 0}
    side = "port"
    for tok in seq:
        if tok == "port" or tok == "stbd":
            side = tok
        elif tok == "load":
            counts[side] += 1
        elif tok == "unload":
            if counts[side] > 0:
                counts[side] -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if counts["port"] > counts["stbd"]:
        return "port"
    if counts["stbd"] > counts["port"]:
        return "stbd"
    return "even"
