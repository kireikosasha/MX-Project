"""Can the sex of every person in a CLUTRR story be read exactly from the text (no weights beyond the parser)?
Evidence: (1) a kin word with a sex that the name stands in apposition to or is compounded with ("her daughter Ann",
"Ann, his son"); (2) he/she/his/her/him/hers in the same sentence when the name is the first named subject of it, or
in the next sentence when it opens with that pronoun. Votes per name; compared with the genders given in the file."""
import csv, re, sys, collections
sys.path.insert(0, sys.argv[1])
import spacy
nlp = spacy.load("en_core_web_sm")
M = {"father", "son", "brother", "husband", "grandfather", "grandson", "uncle", "nephew", "dad", "grandpa", "boy", "man", "he", "his", "him"}
F = {"mother", "daughter", "sister", "wife", "grandmother", "granddaughter", "aunt", "niece", "mom", "mum", "grandma", "girl", "woman", "she", "her", "hers"}
PRON_M, PRON_F = {"he", "his", "him", "himself"}, {"she", "her", "hers", "herself"}
tot = cov = ok = 0
for row in csv.DictReader(open(sys.argv[2], newline="", encoding="utf-8")):
    raw = row["story"]
    text, spans, pos = "", [], 0
    for m in re.finditer(r"\[([^\]]+)\]", raw):
        text += raw[pos:m.start()]; spans.append((len(text), len(text) + len(m.group(1)), m.group(1))); text += m.group(1); pos = m.end()
    text += raw[pos:]
    doc = nlp(text)
    at = {}
    for s, e, name in spans:
        sp = doc.char_span(s, e, alignment_mode="expand")
        if sp is not None:
            at[sp.root.i] = name
    votes = collections.defaultdict(collections.Counter)
    sents = list(doc.sents)
    for si, sent in enumerate(sents):
        named = [t for t in sent if t.i in at]
        for t in named:                                     # apposition / compound with a kin word
            for other in [t.head] + list(t.children):
                if other.dep_ in ("appos", "compound") or t.dep_ in ("appos", "compound"):
                    w = other.lemma_.lower()
                    if w in M - PRON_M: votes[at[t.i]]["m"] += 2
                    if w in F - PRON_F: votes[at[t.i]]["f"] += 2
        subj = next((t for t in sent if t.i in at and t.dep_ in ("nsubj", "nsubjpass")), None)
        if subj is not None:
            others = {at[t.i] for t in named} - {at[subj.i]}
            for t in sent:
                w = t.text.lower()
                if t.i > subj.i and (w in PRON_M or w in PRON_F):
                    votes[at[subj.i]]["m" if w in PRON_M else "f"] += 1
            if si + 1 < len(sents):
                first = sents[si + 1][0].text.lower()
                if first in ("he", "she", "his", "her"):
                    votes[at[subj.i]]["m" if first in ("he", "his") else "f"] += 1
    for g in row["genders"].split(","):
        name, sex = g.split(":")
        if name not in at.values():
            continue
        tot += 1
        if votes[name]:
            cov += 1
            ok += votes[name].most_common(1)[0][0] == ("m" if sex == "male" else "f")
print(f"people named in the stories: {tot}; sex read from the text: {100*cov/tot:.1f}%; right where read: {100*ok/max(1,cov):.1f}%")
