"""A text editor: type x, type y, or undo the last letter typed (nothing to undo: nothing happens); what is the last letter of the text now."""
ALPHABET = ['x', 'y', 'undo']
def answer(seq):
    text = []
    for t in seq:
        if t == "undo":
            if text:
                text.pop()
        else:
            text.append(t)
    return text[-1] if text else "empty"
