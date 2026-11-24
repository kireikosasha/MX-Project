"""a^n b^n: "yes" when the story is some a's followed by as many b's, "maybe" while it can still become one, "no" when it cannot."""
ALPHABET = ['a', 'b']
def answer(seq):
    s = "".join(seq)
    k = len(s) - len(s.lstrip("a"))
    rest = s[k:]
    if rest.strip("b"):
        return "no"
    m = len(rest)
    return "yes" if m == k else "maybe" if m < k else "no"
