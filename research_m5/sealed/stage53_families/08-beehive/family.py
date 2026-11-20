"""Beehive honey keeper.

The hive starts calm with 2 frames of honey; honey is always kept between
0 and 5 frames (extra is lost, it never goes below 0). Read the tokens in
order: "nectar" adds 1 frame if the hive is calm, but does nothing if the
hive is alert (foragers stay home to guard); "wasp" makes a calm hive alert,
and if the hive is already alert the wasp robs 1 frame; "smoke" makes the
hive calm (honey unchanged); "rain" keeps the bees inside, where they eat
1 frame (mood unchanged). The answer is the final mood and honey level,
written as "<mood>:<frames>", e.g. "calm:3" or "alert:0".
"""

ALPHABET = ["nectar", "wasp", "smoke", "rain"]


def answer(seq):
    calm = True
    honey = 2
    for tok in seq:
        if tok == "nectar":
            if calm:
                honey = min(5, honey + 1)
        elif tok == "wasp":
            if calm:
                calm = False
            else:
                honey = max(0, honey - 1)
        elif tok == "smoke":
            calm = True
        elif tok == "rain":
            honey = max(0, honey - 1)
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%s:%d" % ("calm" if calm else "alert", honey)
