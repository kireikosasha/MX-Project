ALPHABET = ['pax', 'call', 'bag', 'load']


def answer(seq):
    shown = 'even'  # what the display shows
    p0 = 0  # a circle of 5
    p1 = 0  # a level from 0 to 4
    p2 = 0  # a level from 0 to 4
    p3 = 0  # a circle of 12
    p4 = 0  # a level from 0 to 6
    p5 = []  # a line, first come first served
    for tok in seq:
        if tok == 'pax':
            effect = {('queue', 0): None, ('belt', 0): 'even', ('belt', 1): None}.get((shown, int(p0 == 4),), 'queue')
            new = shown if effect is None else effect
            seen = int(new != shown)
            d0, s0 = 0, None
            if seen == 1: d0 += 1
            n0 = s0 if s0 is not None else (p0 + d0) % 5
            d1, s1 = 0, None
            if seen == 0: d1 += 3
            n1 = s1 if s1 is not None else p1 + d1 if 0 <= p1 + d1 <= 4 else p1
            d3, s3 = 0, None
            if seen == 0: d3 += 2
            n3 = s3 if s3 is not None else (p3 + d3) % 12
            d4, s4 = 0, None
            if seen == 0: d4 += -3
            n4 = s4 if s4 is not None else p4 + d4 if 0 <= p4 + d4 <= 6 else p4
            n5 = p5
            if seen == 0: n5 = p5 + ['pax']
            p0 = n0
            p1 = n1
            p3 = n3
            p4 = n4
            p5 = n5
            shown = new
        elif tok == 'call':
            effect = {('queue', 0, 1): 'even', ('queue', 0, 0): 'even', ('even', 4, 1): 'belt', ('queue', 1, 1): 'even', ('even', 4, 0): 'belt', ('queue', 4, 0): 'even', ('queue', 2, 1): 'even', ('even', 2, 1): 'belt', ('even', 2, 0): 'belt'}.get((shown, p1, int(p2 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n2 = max(0, p2 - 1)
            d3, s3 = 0, None
            if seen == 1: d3 += 3
            n3 = s3 if s3 is not None else (p3 + d3) % 12
            p2 = n2
            p3 = n3
            shown = new
        elif tok == 'bag':
            effect = {('even', 3, 0): 'belt', ('even', 3, 2): 'belt', ('queue', 4, 2): 'even', ('queue', 6, 1): 'even', ('queue', 5, 2): 'even', ('queue', 9, 1): 'even', ('even', 9, 3): 'belt', ('even', 9, 5): 'belt', ('queue', 6, 6): 'even', ('queue', 11, 5): 'even', ('even', 11, 5): 'belt', ('even', 11, 2): 'belt', ('queue', 1, 4): 'even', ('queue', 1, 6): 'even', ('queue', 8, 6): 'even', ('even', 8, 6): 'belt'}.get((shown, p3, p4,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d1, s1 = 0, None
            if seen == 1: d1 += 1
            n1 = s1 if s1 is not None else p1 + d1 if 0 <= p1 + d1 <= 4 else p1
            d2, s2 = 0, None
            if seen == 0: d2 += 3
            n2 = s2 if s2 is not None else max(0, min(4, p2 + d2))
            n4 = p4 + 2 if 0 <= p4 + 2 <= 6 else p4
            n5 = p5
            if seen == 1: n5 = p5 + ['bag']
            p1 = n1
            p2 = n2
            p4 = n4
            p5 = n5
            shown = new
        elif tok == 'load':
            effect = {('belt', 1): 'even', ('even', 2): 'queue', ('belt', 0): 'even'}.get((shown, ({'pax': 1, 'bag': 2}[p5[0]] if p5 else 0),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d1, s1 = 0, None
            if seen == 1: d1 += -3
            n1 = s1 if s1 is not None else p1 + d1 if 0 <= p1 + d1 <= 4 else p1
            d3, s3 = 0, None
            if seen == 1: d3 += 3
            n3 = s3 if s3 is not None else (p3 + d3) % 12
            n5 = p5[1:]
            p1 = n1
            p3 = n3
            p5 = n5
            shown = new
    return str(shown)
