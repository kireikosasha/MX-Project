"""Recycling-plant baler.

Items arrive one by one at a baler that holds only one material at a time.
Each "paper", "can" or "bottle" goes into the hopper: if the hopper already
holds that same material the item is added to it, otherwise the partial load
is sent back to the sorting pile and the hopper starts fresh with this one
item. A "rag" is a contaminant: it fouls the hopper, which is emptied (the
partial load is lost) and left empty. As soon as the hopper holds 3 items of
the same material it is pressed into a bale, shipped, and the hopper becomes
empty again. The answer is the material of the most recently shipped bale,
or "none" if no bale has been shipped by the end of the sequence.
"""

ALPHABET = ["paper", "can", "bottle", "rag"]

BALE_SIZE = 3


def answer(seq):
    material = None   # material currently in the hopper (None = empty)
    count = 0         # number of items in the hopper
    last_bale = "none"
    for tok in seq:
        if tok == "rag":
            material, count = None, 0
            continue
        if tok == material:
            count += 1
        else:
            material, count = tok, 1
        if count == BALE_SIZE:
            last_bale = material
            material, count = None, 0
    return last_bale
