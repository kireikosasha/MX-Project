"""A second set of machines from outside: the Mealy models of the automata wiki of Radboud University
(https://automata.cs.ru.nl, benchmarks contributed by the authors of the papers named in their titles).

    python external/wiki_fetch.py RAW_FOLDER OUT_FOLDER

fetches the overview and the .dot models of every benchmark (as text; nothing of it is run), selects by the rule named
before any run (README, stage 77) and writes OUT_FOLDER/NN-name/family.py. The models themselves are not kept in this
repository; wiki_selected.txt lists the files selected, their sizes and the hashes of the files as fetched.

The rule: complete deterministic Mealy machines of 2 to 10 states and at most 12 inputs; every benchmark of the
overview but the toy and random ones and the two the first external set was drawn from wholly (the Bluetooth devices
and the TCP stacks); no model whose table is that of a model already run; at most 12 a benchmark, the first by file name.
"""
import glob, hashlib, importlib.util, os, re, subprocess, sys, time, urllib.parse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mealy_dot

SKIP = ("BenchmarkToy", "BenchmarkRandom")
WHOLLY = ("BenchmarkBLE-PferscherEtAl2022", "BenchmarkTCP-Fiterau-BrosteanEtAl2016")


def get(url, path):
    return subprocess.run(["curl", "-sS", "-m", "60", "--max-filesize", "400000", "-o", path, url]).returncode


def fetch(raw):
    os.makedirs(raw, exist_ok=True)
    over = os.path.join(raw, "overview.html")
    if not os.path.exists(over):
        get("https://automata.cs.ru.nl/Overview", over)
    text = open(over, encoding="utf-8", errors="replace").read()
    for b in sorted(set(re.findall(r"automata\.cs\.ru\.nl/(Benchmark[A-Za-z0-9_-]+)/Description", text))):
        if b.startswith(SKIP):
            continue
        d = os.path.join(raw, b)
        os.makedirs(d, exist_ok=True)
        page = os.path.join(d, "_mealy.html")
        if not os.path.exists(page):
            get(f"https://automata.cs.ru.nl/{b}/Mealy", page)
            time.sleep(0.5)
        text_b = open(page, encoding="utf-8", errors="replace").read() if os.path.exists(page) else ""
        links = sorted(set(re.findall(r"href='(https://automata\.cs\.ru\.nl/pmwiki/uploads/" + re.escape(b) + r"/[^']+\.dot)'", text_b)))
        for u in links[:60]:
            f = os.path.join(d, urllib.parse.unquote(u.rsplit("/", 1)[1]).replace("/", "_"))
            if not os.path.exists(f):
                get(u, f)
                time.sleep(0.3)


def select(raw):
    ran = set()
    for d in glob.glob(os.path.join(HERE, "protocols", "*", "family.py")):
        spec = importlib.util.spec_from_file_location("family", d)
        fam = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fam)
        ran.add(frozenset(fam.TRANS.items()))
    chosen = []
    for b in sorted(os.listdir(raw)):
        if not os.path.isdir(os.path.join(raw, b)) or b in WHOLLY:
            continue
        ok = []
        for f in sorted(glob.glob(os.path.join(raw, b, "*.dot"))):
            try:
                inputs, start, trans = mealy_dot.read(f)
            except Exception:
                continue
            if not trans or not inputs:
                continue
            states = len(mealy_dot.reached(start, inputs, trans))
            if not 2 <= states <= 10 or len(inputs) > 12 or frozenset(trans.items()) in ran:
                continue
            ok.append((b, os.path.basename(f), states, len(inputs)))
        chosen += ok[:12]
    return chosen


if __name__ == "__main__":
    raw, out = sys.argv[1], sys.argv[2]
    fetch(raw)
    lines = []
    for n, (b, f, states, inputs) in enumerate(select(raw), 1):
        path = os.path.join(raw, b, f)
        ins, start, trans = mealy_dot.read(path)
        short = re.sub(r"^Benchmark", "", b).split("-")[0].lower()
        name = f"{n:02d}-{short}-" + re.sub(r"[^a-z0-9]+", "-", f[:-4].lower()).strip("-")[:40]
        os.makedirs(os.path.join(out, name), exist_ok=True)
        body = f'"""{b}/{f}: a Mealy machine of the automata wiki (automata.cs.ru.nl). {states} states, {inputs} inputs.\n' \
               f'The answer to a story of inputs is the output to the last of them."""\n\n' \
               f"ALPHABET = {ins!r}\nSTART = {start!r}\nTRANS = {{\n" + \
               "".join(f"    {k!r}: {v!r},\n" for k, v in sorted(trans.items())) + "}\n\n\n" \
               "def answer(seq):\n    q, out = START, \"\"\n    for a in seq:\n        q, out = TRANS[q, a]\n    return out\n"
        open(os.path.join(out, name, "family.py"), "w").write(body)
        lines.append(f"{name}\t{b}/{f}\t{states} states\t{inputs} inputs\t{hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
    open(os.path.join(out, "selected.txt"), "w").write("\n".join(lines) + "\n")
    print(len(lines), "families written to", out)
