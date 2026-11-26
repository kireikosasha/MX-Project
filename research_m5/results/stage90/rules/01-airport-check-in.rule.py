ALPHABET = ['pax', 'call', 'bag', 'load']


def answer(seq):
    shown = 'even'  # what the display shows
    p0 = 0  # a circle of 7
    p1 = 0  # a level from 0 to 5
    p2 = 0  # a counter, not below zero
    p3 = 0  # a level from 0 to 6
    p4 = 0  # a counter, not below zero
    p5 = 0  # a level from 0 to 4
    for tok in seq:
        if tok == 'pax':
            effect = {('even', 0): 'queue', ('belt', 0): 'even'}.get((shown, int(p0 == 6),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n1 = min(5, p1 + 1)
            d3, s3 = 0, None
            if seen == 0: d3 += 2
            n3 = s3 if s3 is not None else max(0, min(6, p3 + d3))
            n5 = p5 + 2 if 0 <= p5 + 2 <= 4 else p5
            p1 = n1
            p3 = n3
            p5 = n5
            shown = new
        elif tok == 'call':
            effect = {('queue', 1, 0): 'even', ('even', 2, 1): 'belt', ('queue', 2, 1): 'even', ('even', 1, 1): 'belt', ('queue', 3, 1): 'even'}.get((shown, p1, int(p2 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n1 = max(0, p1 - 1)
            d3, s3 = 0, None
            if seen == 0: d3 += -2
            if seen == 1: d3 += 3
            n3 = s3 if s3 is not None else max(0, min(6, p3 + d3))
            p1 = n1
            p3 = n3
            shown = new
        elif tok == 'bag':
            effect = {('even', 3, 0): 'belt', ('even', 3, 1): 'belt', ('queue', 2, 1): 'even', ('queue', 4, 0): 'even', ('queue', 5, 1): 'even', ('even', 5, 1): 'belt', ('queue', 3, 0): 'even', ('queue', 1, 1): 'even', ('even', 6, 0): 'belt', ('even', 6, 1): 'belt'}.get((shown, p3, int(p4 >= 1),), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            d0, s0 = 0, None
            if seen == 1: d0 += -2
            n0 = s0 if s0 is not None else (p0 + d0) % 7
            d2, s2 = 0, None
            if seen == 1: d2 += 2
            n2 = s2 if s2 is not None else max(0, min(10 ** 9, p2 + d2))
            d4, s4 = 0, None
            if seen == 0: d4 += 1
            n4 = s4 if s4 is not None else max(0, min(10 ** 9, p4 + d4))
            p0 = n0
            p2 = n2
            p4 = n4
            shown = new
        elif tok == 'load':
            effect = {('belt', 3): 'even', ('even', 4): 'queue', ('belt', 1): 'even', ('belt', 4): 'even', ('even', 3): 'queue'}.get((shown, p5,), None)
            new = shown if effect is None else effect
            seen = int(new != shown)
            n2 = max(0, p2 - 1)
            d4, s4 = 0, None
            if seen == 0: s4 = 0
            n4 = s4 if s4 is not None else max(0, min(10 ** 9, p4 + d4))
            n5 = p5 - 1 if 0 <= p5 - 1 <= 4 else p5
            p2 = n2
            p4 = n4
            p5 = n5
            shown = new
    return str(shown)
