"""Bluetooth/bluetooth_model.dot: a Mealy machine learned by others from a real implementation (AALpy, DotModels, MIT licence).
3 states, 9 inputs. The answer to a story of inputs is the output to the last of them."""

ALPHABET = ['i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8']
START = 's0'
TRANS = {
    ('s0', 'i0'): ('s0', 'o0'),
    ('s0', 'i1'): ('s1', 'o1'),
    ('s0', 'i2'): ('s0', 'o2'),
    ('s0', 'i3'): ('s0', 'o2'),
    ('s0', 'i4'): ('s0', 'o2'),
    ('s0', 'i5'): ('s0', 'o2'),
    ('s0', 'i6'): ('s0', 'o2'),
    ('s0', 'i7'): ('s0', 'o2'),
    ('s0', 'i8'): ('s0', 'o2'),
    ('s1', 'i0'): ('s0', 'o0'),
    ('s1', 'i1'): ('s1', 'o1'),
    ('s1', 'i2'): ('s1', 'o3'),
    ('s1', 'i3'): ('s1', 'o3'),
    ('s1', 'i4'): ('s1', 'o4'),
    ('s1', 'i5'): ('s1', 'o5'),
    ('s1', 'i6'): ('s2', 'o6'),
    ('s1', 'i7'): ('s1', 'o7'),
    ('s1', 'i8'): ('s1', 'o8'),
    ('s2', 'i0'): ('s0', 'o0'),
    ('s2', 'i1'): ('s1', 'o1'),
    ('s2', 'i2'): ('s2', 'o3'),
    ('s2', 'i3'): ('s2', 'o3'),
    ('s2', 'i4'): ('s2', 'o4'),
    ('s2', 'i5'): ('s2', 'o5'),
    ('s2', 'i6'): ('s2', 'o1'),
    ('s2', 'i7'): ('s2', 'o7'),
    ('s2', 'i8'): ('s2', 'o8'),
}


def answer(seq):
    q, out = START, ""
    for a in seq:
        q, out = TRANS[q, a]
    return out
