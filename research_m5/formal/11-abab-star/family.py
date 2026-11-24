"""(abab)* (Bhattamishra, Ahuja, Goyal 2020), over the letters a and b: "yes" when the story is 'abab' written some
number of times, "maybe" while it can still become that, "no" once it cannot."""
ALPHABET = ['a', 'b']
WORD = 'abab'
def answer(seq):
    for i, t in enumerate(seq):
        if t != WORD[i % len(WORD)]:
            return "no"
    return "yes" if len(seq) % len(WORD) == 0 else "maybe"
