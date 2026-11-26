"""Orchid nursery bench.

The greenhouse air is dry, fair or humid and starts fair; the sales bench
starts empty. Read the tokens in order: "pot" puts one new orchid on the
bench; "mist" makes the air one step damper (dry to fair, fair to humid,
humid stays humid); "sun" makes the air one step drier (humid to fair, fair
to dry), but if the air is already dry the air stays dry and instead one
orchid on the bench wilts and is thrown out (nothing happens if the bench is
empty); "buyer" takes one orchid off the bench, but only if the air is not
dry and the bench is not empty (in dry air the flowers look poor and the
buyer leaves with nothing). The answer is the number of orchids left on the
bench after the last token, written as a decimal number such as "0" or "7".
"""

ALPHABET = ["pot", "mist", "sun", "buyer"]


def answer(seq):
    humidity = 1  # 0 = dry, 1 = fair, 2 = humid
    bench = 0
    for tok in seq:
        if tok == "pot":
            bench += 1
        elif tok == "mist":
            if humidity < 2:
                humidity += 1
        elif tok == "sun":
            if humidity > 0:
                humidity -= 1
            elif bench > 0:
                bench -= 1
        elif tok == "buyer":
            if humidity > 0 and bench > 0:
                bench -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return str(bench)
