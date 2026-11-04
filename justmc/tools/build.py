#!/usr/bin/env python3
"""
Build every JustMC module of the port with JMCC and sanity-check the result.

    python build.py --jmcc /path/to/JMCC        (folder with jmcc.py, compilator.py, data/)

Outputs (justmc/build/):
    millennium_trainer.json   library + "~ml" admin commands (pure ML world)
    millennium_mx_demo.json   library + admin + MX aim-check demo
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


def postprocess(path):
    """Icons and English + Russian descriptions for every function / process (tools/function_meta.py).
    Every function must fit in one JustMC code line: helper functions "jmcc.N" are not allowed."""
    code = json.load(open(path))
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
    for src, out in TARGETS.items():
        s = os.path.abspath(os.path.join(SRC, src))
        o = os.path.abspath(os.path.join(OUT, out))
        if os.path.exists(o):
            os.remove(o)
        print(f"== {src} -> build/{out}")
        if not compile_one(jmcc, s, o):
            ok = False
            continue
        ok = postprocess(o) and ok
        r = subprocess.run([sys.executable, os.path.join(HERE, "check_build.py"), o])
        ok = ok and r.returncode == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
