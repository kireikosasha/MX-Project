"""Dog show ring: ribbons for poodles versus terriers.

Dogs wait in a single line at the edge of the ring, which starts empty. Reading the
tokens in order: "poodle" or "terrier" means a dog of that breed joins the back of
the line; "bark" means the dog at the front of the line barks, so the steward sends
it to the back of the line (nothing happens if the line is empty); "ribbon" means the
judge gives a ribbon to the dog at the front of the line, which then leaves the ring
for good (if the line is empty, the ribbon goes unawarded and nothing happens). At
the end, the answer is the breed that has won more ribbons in total, "poodle" or
"terrier", or "tie" if both breeds have won the same number (including none).
"""

from collections import deque

ALPHABET = ["poodle", "terrier", "bark", "ribbon"]


def answer(seq):
    line = deque()
    lead = 0  # poodle ribbons minus terrier ribbons
    for tok in seq:
        if tok == "poodle" or tok == "terrier":
            line.append(tok)
        elif tok == "bark":
            if line:
                line.append(line.popleft())
        elif tok == "ribbon":
            if line:
                dog = line.popleft()
                lead += 1 if dog == "poodle" else -1
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if lead > 0:
        return "poodle"
    if lead < 0:
        return "terrier"
    return "tie"
