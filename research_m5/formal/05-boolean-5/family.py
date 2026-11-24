"""Boolean-5 (Bhattamishra, Ahuja, Goyal 2020): well-formed Boolean expressions in prefix (Polish) notation over the
values 0 and 1 and 5 operators, ~ of 1 argument, & of 2 arguments, | of 2 arguments, ^ of 2 arguments, ? of 3 arguments. "yes" when the
story is one whole expression, "maybe" while it can still become one, "no" once a symbol comes after the expression ended."""
ALPHABET = ['0', '1', '~', '&', '|', '^', '?']
ARITY = {'~': 1, '&': 2, '|': 2, '^': 2, '?': 3}
def answer(seq):
    need = 1
    for t in seq:
        if need == 0:
            return "no"
        need += ARITY.get(t, 0) - 1
    return "yes" if need == 0 else "maybe"
