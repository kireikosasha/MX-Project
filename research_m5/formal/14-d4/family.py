"""D_4 (Bhattamishra, Ahuja, Goyal 2020; dot-depth 4, star-free): D_1 = (ab)*, D_k = (a D_(k-1) b)*, that is
a and b as brackets, balanced and never deeper than 4. "yes" when balanced, "maybe" while it can still become so, "no"
from the first b with nothing open or the first a that goes deeper than 4."""
ALPHABET = ['a', 'b']
DEPTH = 4
def answer(seq):
    d = 0
    for t in seq:
        d += 1 if t == 'a' else -1
        if d < 0 or d > DEPTH:
            return "no"
    return "yes" if d == 0 else "maybe"
