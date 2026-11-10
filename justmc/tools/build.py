#!/usr/bin/env python3
"""
Build every JustMC module of the port with JMCC and sanity-check the result.

    python build.py --jmcc /path/to/JMCC        (folder with jmcc.py, compilator.py, data/)

Outputs (justmc/build/):
    millennium_trainer.json   library + "~ml" admin commands (pure ML world)
    millennium_mx_demo.json   library + admin + MX aim-check demo
    millennium_digits_demo.json  library + admin + digit recognition demo
Upload a .json to any file host (or use `jmcc.py compile <file> -u`) and load it in JustMC.
"""
import json
import os
import re
import subprocess
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from function_meta import META  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
OUT = os.path.join(HERE, "..", "build")
TARGETS = {
    "ml_trainer.jc": "millennium_trainer.json",
    "demo_mx_aim.jc": "millennium_mx_demo.json",
    "demo_digits.jc": "millennium_digits_demo.json",
    "probe_ai.jc": "millennium_probe.json",
    "demo_reader.jc": "millennium-6m-llm.json",
}
# the same module for another chat model: (built file it is made from, text replacements in it)
VARIANTS = {
    "millennium-13m-llm.json": ("millennium-6m-llm.json", [
        ("/justmc/weights/chat/", "/justmc/weights/chat13m/"),
        ("several hundred files, 5-10 minutes", "about 2300 files, 25-35 minutes"),
        ("about a thousand files, 10-15 minutes", "about 2600 files, 30-40 minutes"),
    ]),
}


def compile_one(jmcc_dir, src, out):
    # jmcc.py does os.chdir() with a Windows-only path split, so drive compilator directly
    code = f"""
import os, sys, types
os.chdir({jmcc_dir!r}); sys.path.insert(0, {jmcc_dir!r})
txt = open('jmcc.py', encoding='utf-8').read()
txt = txt[:txt.index("os.system('color')")]
mod = types.ModuleType('jmcc'); mod.__file__ = os.path.join({jmcc_dir!r}, 'jmcc.py')
exec(compile(txt, mod.__file__, 'exec'), mod.__dict__)
data = mod.Properties(text=open('jmcc.properties', encoding='UTF-8').read())
lang = 'data/lang/ru_RU.properties'
mod.lang = mod.Properties(text=open(lang, encoding='UTF-8').read()) if os.path.exists(lang) else None
sys.modules['jmcc'] = mod
from compilator import compile_file
props = data.properties.copy(); props['compact'] = True; props['output'] = {out!r}
compile_file({src!r}, upload=False, properties=props)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode == 0 and os.path.exists(out)


def _text(t):
    return {"type": "text", "text": t, "parsing": "legacy"}


def _wrap(text, width=42):
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines


def block_arguments(jmcc_dir):
    """{action id: {argument id}} of the arguments that take blocks"""
    acts = json.load(open(os.path.join(jmcc_dir, "data", "actions.json"), encoding="utf-8"))
    acts = acts if isinstance(acts, list) else list(acts.values())
    return {a["id"]: {g["id"] for g in a["args"] if g.get("type") == "block"} for a in acts}


def fix_block_lists(ops, spec):
    """JMCC turns a text into a block value only for a single block argument; inside a list of
    blocks (world::block_equals, allow_placing_breaking_blocks, ...) the texts stay texts"""
    n = 0
    for op in ops:
        for v in op.get("values", []):
            if v["name"] in spec.get(op.get("action"), ()) and v["value"].get("type") == "array":
                for i, it in enumerate(v["value"]["values"]):
                    if it.get("type") == "text":
                        bid = it["text"].split(":", 1)[-1] if it["text"].startswith("minecraft:") else it["text"]
                        v["value"]["values"][i] = {"type": "block", "block": bid}
                        n += 1
        n += fix_block_lists(op.get("operations", []), spec)
    return n


def build_stamp():
    """hash of all sources: a module built from other sources sets its cached game variables up again"""
    import hashlib
    h = hashlib.sha1()
    for name in sorted(os.listdir(SRC)):
        if name.endswith(".jc"):
            h.update(name.encode() + b"\0" + open(os.path.join(SRC, name), "rb").read() + b"\0")
    return h.hexdigest()[:10]


def postprocess(path, jmcc_dir):
    """Icons and English + Russian descriptions for every function / process (tools/function_meta.py).
    Every function must fit in one JustMC code line: helper functions "jmcc.N" are not allowed."""
    code = json.load(open(path))
    spec = block_arguments(jmcc_dir)
    for h in code["handlers"]:
        fix_block_lists(h.get("operations", []), spec)
    missing, split = [], []
    for h in code["handlers"]:
        if h["type"] not in ("function", "process"):
            continue
        name = h["name"]
        if name.startswith("jmcc."):
            split.append(name)
            continue
        if name not in META:
            missing.append(name)
            continue
        icon, en, ru = META[name]
        desc = ["&f" + x for x in _wrap(en)] + [" "] + ["&7" + x for x in _wrap(ru)]
        h["values"] = [{"name": "description", "value": {"type": "array", "values": [_text(x) for x in desc]}},
                       {"name": "icon", "value": {"type": "item", "item": '{id:"%s",count:1}' % icon}}]
    text = json.dumps(code, separators=(",", ":"), ensure_ascii=False)
    text = re.sub(r"jmcc\.(\d+)", r"tmp.\1", text)   # temporary variables of the compiler
    text = text.replace("MLBUILDSTAMP", build_stamp())   # ML_BUILD: game variables of another build are rebuilt
    open(path, "w", encoding="utf-8").write(text)
    if split:
        print("ERROR: functions longer than one JustMC code line were split by JMCC:", split)
    print(f"postprocess: missing descriptions: {missing}")
    return not missing and not split


def main():
    jmcc = None
    if "--jmcc" in sys.argv:
        jmcc = os.path.abspath(sys.argv[sys.argv.index("--jmcc") + 1])
    jmcc = jmcc or os.environ.get("JMCC_DIR")
    if not jmcc or not os.path.exists(os.path.join(jmcc, "compilator.py")):
        sys.exit("JMCC folder not found: pass --jmcc <dir> or set JMCC_DIR")
    os.makedirs(OUT, exist_ok=True)
    ok = True
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for src, out in TARGETS.items():
        if only and only != src:
            continue
        s = os.path.abspath(os.path.join(SRC, src))
        o = os.path.abspath(os.path.join(OUT, out))
        if os.path.exists(o):
            os.remove(o)
        print(f"== {src} -> build/{out}")
        if not compile_one(jmcc, s, o):
            ok = False
            continue
        ok = postprocess(o, jmcc) and ok
        r = subprocess.run([sys.executable, os.path.join(HERE, "check_build.py"), o])
        ok = ok and r.returncode == 0
        for name, (base, repl) in VARIANTS.items():
            if base == out and os.path.exists(o):
                text = open(o, encoding="utf-8").read()
                for x, y in repl:
                    if x not in text:
                        print(f"ERROR: {name}: text to replace not found: {x}")
                        ok = False
                    text = text.replace(x, y)
                open(os.path.join(OUT, name), "w", encoding="utf-8").write(text)
                print(f"== build/{out} -> build/{name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
