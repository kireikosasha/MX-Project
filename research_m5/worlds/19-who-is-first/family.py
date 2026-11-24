"""A line at a counter: a or b joins the end, serve sends away the one at the front (none: nothing happens); who is at the front now."""
ALPHABET = ['a', 'b', 'serve']
def answer(seq):
    line = []
    for t in seq:
        if t == "serve":
            if line:
                line.pop(0)
        else:
            line.append(t)
    return line[0] if line else "nobody"
