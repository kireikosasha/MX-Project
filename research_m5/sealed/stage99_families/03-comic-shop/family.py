"""Comic shop shelves at closing time.

A comic shop has two shelves, one for the "Hero" series and one for the
"Villain" series, and both start empty. Read the tokens in order: "hero" puts
one new Hero copy on its shelf and "villain" puts one new Villain copy on its
shelf (shelves have no size limit); "kid" is a kid who buys one Hero copy if
there is any, otherwise one Villain copy if there is any, otherwise leaves with
nothing; "collector" buys one Villain copy if there is any and otherwise leaves
with nothing (collectors never buy Hero). At closing time (after the last
token) the answer is "empty" if both shelves are empty, "tie" if they hold the
same positive number of copies, and otherwise the name of the shelf holding
more copies, "hero" or "villain".
"""

ALPHABET = ["hero", "villain", "kid", "collector"]


def answer(seq):
    hero = 0
    villain = 0
    for tok in seq:
        if tok == "hero":
            hero += 1
        elif tok == "villain":
            villain += 1
        elif tok == "kid":
            if hero > 0:
                hero -= 1
            elif villain > 0:
                villain -= 1
        elif tok == "collector":
            if villain > 0:
                villain -= 1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if hero == 0 and villain == 0:
        return "empty"
    if hero == villain:
        return "tie"
    return "hero" if hero > villain else "villain"
