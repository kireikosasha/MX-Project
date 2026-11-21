"""Subway turnstile with a station alarm.

The station starts with the turnstile gate shut, no guard on duty and the alarm
level at 0 (levels 0..3 are named calm, uneasy, tense, alarm). Tokens are read in
order: "tap" (a rider taps a fare card) opens the gate, and it simply stays open
if it already was; "push" (a rider pushes the bar) does nothing if the gate is
shut, but if the gate is open the rider passes, the gate shuts again behind them
and the alarm drops by one level (never below calm), because paying traffic
calms the station; "hop" (someone jumps the turnstile) resets the alarm to calm
if a guard is on duty (the hopper is caught), and otherwise raises the alarm by
one level (never above alarm); "guard" toggles the guard between on duty and off
duty. The gate, guard and alarm are not affected by any other token. The answer
is the final alarm level name and gate state joined by a dash, for example
"tense-open" or "calm-shut".
"""

ALPHABET = ["tap", "push", "hop", "guard"]

_LEVELS = ["calm", "uneasy", "tense", "alarm"]


def answer(seq):
    gate_open = False
    guard = False
    level = 0
    for tok in seq:
        if tok == "tap":
            gate_open = True
        elif tok == "push":
            if gate_open:
                gate_open = False
                if level > 0:
                    level -= 1
        elif tok == "hop":
            if guard:
                level = 0
            elif level < 3:
                level += 1
        elif tok == "guard":
            guard = not guard
        else:
            raise ValueError("unknown token: %r" % (tok,))
    return _LEVELS[level] + "-" + ("open" if gate_open else "shut")
