ALPHABET = ['urgent', 'minor', 'doctor', 'nurse']


def answer(seq):
    shown = 'empty'  # what the display shows
    p0 = 0  # a level from 0 to 6
    p1 = 0  # a whole number
    p2 = 0  # a level from 0 to 6
    p3 = 0  # a level from 0 to 6
    p4 = 0  # a level from 0 to 6
    p5 = 0  # a circle of 12
    p6 = 0  # a level from 0 to 5
    p7 = 0  # a level from 0 to 5
    for tok in seq:
        if tok == 'urgent':
            effect = {('empty', 0, 0): 'urgent', ('minor', 0, 0): 'tie', ('tie', 0, 0): 'urgent', ('minor', 0, 1): 'tie', ('tie', 2, 1): 'urgent', ('tie', 3, 1): 'urgent', ('tie', 0, 1): 'urgent', ('minor', 1, 0): 'tie', ('minor', 2, 1): 'tie'}.get((shown, p0, int(p1 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d1, s1 = 0, None
            if seen == 0: d1 += 1
            n1 = s1 if s1 is not None else p1 + d1
            d2, s2 = 0, None
            if seen == 0: d2 += 3
            n2 = s2 if s2 is not None else max(0, min(6, p2 + d2))
            d3, s3 = 0, None
            if seen == 0: d3 += 3
            n3 = s3 if s3 is not None else p3 + d3 if 0 <= p3 + d3 <= 6 else p3
            n4 = min(6, p4 + 1)
            d5, s5 = 0, None
            if seen == 1: d5 += 3
            n5 = s5 if s5 is not None else (p5 + d5) % 12
            d6, s6 = 0, None
            if seen == 1: s6 = 5
            n6 = s6 if s6 is not None else p6 + d6 if 0 <= p6 + d6 <= 5 else p6
            p1 = n1
            p2 = n2
            p3 = n3
            p4 = n4
            p5 = n5
            p6 = n6
            shown = new
        elif tok == 'minor':
            effect = {('empty', 0, 1): 'minor', ('empty', 0, 0): 'minor', ('urgent', 0, 0): 'tie', ('tie', 0, 0): 'minor', ('urgent', 0, 1): 'tie', ('empty', 0, 2): 'minor', ('tie', 0, 1): 'minor', ('tie', 0, 2): 'minor', ('urgent', 2, 3): 'tie', ('urgent', 1, 2): 'tie', ('urgent', 0, 2): 'tie', ('empty', 0, 3): 'minor', ('tie', 1, 3): 'minor', ('tie', 0, 3): 'minor', ('tie', 3, 3): 'minor', ('urgent', 4, 6): 'tie', ('urgent', 2, 6): 'tie', ('urgent', 1, 3): 'tie'}.get((shown, p2, p3,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d0, s0 = 0, None
            if seen == 0: d0 += 2
            n0 = s0 if s0 is not None else max(0, min(6, p0 + d0))
            n2 = max(0, p2 - 1)
            d5, s5 = 0, None
            if seen == 0: d5 += -1
            n5 = s5 if s5 is not None else (p5 + d5) % 12
            d6, s6 = 0, None
            if seen == 0: d6 += 2
            n6 = s6 if s6 is not None else p6 + d6 if 0 <= p6 + d6 <= 5 else p6
            d7, s7 = 0, None
            if seen == 0: d7 += 2
            n7 = s7 if s7 is not None else max(0, min(5, p7 + d7))
            p0 = n0
            p2 = n2
            p5 = n5
            p6 = n6
            p7 = n7
            shown = new
        elif tok == 'doctor':
            effect = {('urgent', 1, 0): 'empty', ('urgent', 1, 3): 'empty', ('minor', 0, 0): 'empty', ('tie', 1, 3): 'minor', ('minor', 0, 3): 'empty', ('minor', 0, 9): 'empty', ('urgent', 1, 9): 'empty', ('tie', 1, 0): 'minor', ('urgent', 2, 2): 'tie', ('urgent', 6, 9): 'tie', ('tie', 1, 9): 'minor', ('urgent', 3, 3): 'tie', ('tie', 1, 2): 'minor', ('minor', 0, 2): 'empty', ('urgent', 2, 6): 'tie', ('urgent', 3, 8): 'tie', ('tie', 2, 8): 'minor', ('tie', 2, 2): 'minor', ('urgent', 4, 9): 'tie', ('tie', 3, 6): 'minor', ('tie', 2, 3): 'minor'}.get((shown, p4, p5,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d0, s0 = 0, None
            if seen == 0: d0 += -1
            n0 = s0 if s0 is not None else max(0, min(6, p0 + d0))
            n2 = max(0, p2 - 2)
            n3 = p3 - 3 if 0 <= p3 - 3 <= 6 else p3
            n4 = max(0, p4 - 1)
            d6, s6 = 0, None
            if seen == 1: d6 += -3
            n6 = s6 if s6 is not None else p6 + d6 if 0 <= p6 + d6 <= 5 else p6
            p0 = n0
            p2 = n2
            p3 = n3
            p4 = n4
            p6 = n6
            shown = new
        elif tok == 'nurse':
            effect = {('minor', 0, 0): 'empty', ('tie', 5, 0): 'urgent', ('minor', 5, 0): 'tie', ('minor', 2, 0): 'empty', ('minor', 4, 0): 'empty', ('tie', 2, 0): 'urgent', ('tie', 5, 2): 'urgent', ('minor', 5, 2): 'tie', ('tie', 5, 4): 'urgent', ('minor', 2, 3): 'tie', ('minor', 4, 1): 'tie'}.get((shown, p6, p7,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n0 = max(0, p0 - 2)
            d1, s1 = 0, None
            if seen == 1: d1 += -3
            n1 = s1 if s1 is not None else p1 + d1
            d3, s3 = 0, None
            if seen == 0: d3 += 1
            n3 = s3 if s3 is not None else p3 + d3 if 0 <= p3 + d3 <= 6 else p3
            n5 = (p5 - 3) % 12
            d7, s7 = 0, None
            if seen == 0: d7 += -3
            n7 = s7 if s7 is not None else max(0, min(5, p7 + d7))
            p0 = n0
            p1 = n1
            p3 = n3
            p5 = n5
            p7 = n7
            shown = new
    return str(shown)
