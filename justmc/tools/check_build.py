#!/usr/bin/env python3
"""Sanity check of a JMCC build: every action argument must be a plain value.
(JMCC 0.5.x sometimes leaves an un-flattened action inside an argument when expressions
are nested 3+ levels deep — such code would not load on JustMC.)"""
import json
import sys

bad = 0
code = json.load(open(sys.argv[1]))


def walk(ops, where):
    global bad
    for op in ops:
        for v in op.get("values", []):
            val = v["value"]
            stack = [val]
            while stack:
                x = stack.pop()
                if "type" not in x:
                    bad += 1
                    print(f"[{where}] {op['action']}: argument '{v['name']}' is not a value")
                    break
                if x["type"] == "array":
                    stack.extend(x["values"])
        walk(op.get("operations", []), where)


ops_total = 0
for h in code["handlers"]:
    walk(h["operations"], h.get("name", h.get("event")))


def count(ops):
    n = 0
    for op in ops:
        n += 1 + count(op.get("operations", []))
    return n


for h in code["handlers"]:
    ops_total += count(h["operations"])
# Control flow that must not cross a JMCC line split: JMCC moves overflowing blocks into helper
# functions "jmcc.N"; a return / stop_repeat / skip_iteration inside such a helper would only
# leave the helper.
split_bad = 0


def walk_ctrl(ops, where, depth):
    global split_bad
    for op in ops:
        if op["action"] in ("control_return_function", "control_stop_repeat", "control_skip_iteration"):
            if where.startswith("jmcc."):
                split_bad += 1
                print(f"[{where}] {op['action']} inside a JMCC split helper")
        walk_ctrl(op.get("operations", []), where, depth + 1)


for h in code["handlers"]:
    if h["type"] == "function":
        walk_ctrl(h["operations"], h.get("name", ""), 0)


# every "else" must directly follow an if_* block (a split must not separate them)
def walk_else(ops, where):
    global split_bad
    prev = None
    for op in ops:
        if op["action"] == "else" and (prev is None or not prev.startswith("if_")):
            split_bad += 1
            print(f"[{where}] 'else' without a preceding if")
        prev = op["action"]
        walk_else(op.get("operations", []), where)


for h in code["handlers"]:
    walk_else(h["operations"], h.get("name", h.get("event", "")))
print(f"handlers: {len(code['handlers'])}, actions: {ops_total}, malformed arguments: {bad}, control flow across splits: {split_bad}")
sys.exit(1 if bad or split_bad else 0)
