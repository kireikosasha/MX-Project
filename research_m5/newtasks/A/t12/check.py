import re, subprocess, sys
G = sys.argv[1]
out = subprocess.run([sys.executable, G, "2000", "7"], capture_output=True, text=True, check=True).stdout
lines = out.rstrip("\n").split("\n")
stories, cur = [], []
expect = 1
words = set()
for ln in lines:
    m = re.match(r"^(\d+) (.*)$", ln)
    assert m, ln
    num, rest = int(m.group(1)), m.group(2)
    if num == 1:
        if cur: stories.append(cur)
        cur = []
    assert num == len(cur) + 1, ("numbering", ln)
    if "\t" in rest:
        q, a = rest.split("\t")
        assert q.endswith("?") and q.count("?") == 1 and "\t" not in a, ln
        assert re.fullmatch(r"[a-z]+|yes|no", a), ln
        cur.append(("Q", q, a))
    else:
        assert rest.endswith(".") and "?" not in rest, ln
        cur.append(("S", rest, None))
    assert re.fullmatch(r"[A-Za-z0-9 .?]+", rest.split("\t")[0]), ln
    words.update(w.lower() for w in re.findall(r"[A-Za-z]+", rest))
stories.append(cur)
bad = 0
for st in stories:
    nq = sum(1 for t in st if t[0] == "Q")
    ns = len(st) - nq
    assert 1 <= nq <= 5 and 3 <= ns <= 12 and 4 <= len(st) <= 12, (nq, ns)
    holder = {}
    for kind, text, ans in st:
        if kind == "S":
            m = re.fullmatch(r"(\w+) (?:holds|has) the (\w+)\.", text)
            if m:
                assert m.group(1) not in holder
                assert m.group(2) not in holder.values()
                holder[m.group(1)] = m.group(2); continue
            m = re.fullmatch(r"(\w+) and (\w+) (?:swap their things|trade what they hold)\.", text)
            assert m and m.group(1) in holder and m.group(2) in holder, text
            a, b = m.group(1), m.group(2)
            holder[a], holder[b] = holder[b], holder[a]
        else:
            m = re.fullmatch(r"What does (\w+) hold\?", text)
            if m: exp = holder[m.group(1)]
            else:
                m = re.fullmatch(r"Who holds the (\w+)\?", text)
                if m: exp = [p for p in holder if holder[p] == m.group(1)][0].lower()
                else:
                    m = re.fullmatch(r"Does (\w+) hold the (\w+)\?", text)
                    assert m, text
                    exp = "yes" if holder[m.group(1)] == m.group(2) else "no"
            if exp != ans:
                bad += 1
print("stories", len(stories), "bad answers", bad)
print("vocab size", len(words), sorted(words))
