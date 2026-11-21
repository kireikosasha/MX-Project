"""Beach patrol: the warning flag and the swimmers in the roped swim zone.

The lifeguard tower starts the day with the green flag up and nobody in the
water. Reading the log token by token: "gust" raises the flag one step
(green -> yellow -> red, staying at red), "calm" lowers it one step (red ->
yellow -> green, staying at green), and "rip" puts the red flag up at once.
"swimmer" means someone wants to go in: if the flag is red they are turned
away, otherwise they enter the swim zone unless it already holds 4 swimmers
(the most it allows), in which case they are also turned away. "whistle" means
the guard calls swimmers out: under red everyone leaves the water, under
yellow one swimmer leaves (if anyone is in), and under green nothing happens.
The answer is the flag colour and the number of swimmers in the zone at the
end of the log, written as "colour/count", for example "yellow/2".
"""

ALPHABET = ["gust", "calm", "rip", "swimmer", "whistle"]

_FLAGS = ["green", "yellow", "red"]
_ZONE_CAPACITY = 4


def answer(seq):
    flag = 0  # index into _FLAGS
    crowd = 0
    for tok in seq:
        if tok == "gust":
            flag = min(2, flag + 1)
        elif tok == "calm":
            flag = max(0, flag - 1)
        elif tok == "rip":
            flag = 2
        elif tok == "swimmer":
            if flag != 2 and crowd < _ZONE_CAPACITY:
                crowd += 1
        elif tok == "whistle":
            if flag == 2:
                crowd = 0
            elif flag == 1 and crowd > 0:
                crowd -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return "%s/%d" % (_FLAGS[flag], crowd)
