"""Machines of real protocol implementations as families for sealed_run.py.

The models are those of the AALpy collection (https://github.com/DES-Lab/AALpy, DotModels, MIT licence): Mealy machines
that others learned from real Bluetooth Low Energy devices, TLS servers, MQTT brokers and TCP stacks. None was written
for this project. A family made of a model: the alphabet is the inputs of the machine, and the answer to a story of
inputs is the output the implementation gives to the last of them.

    python external/mealy_dot.py PATH_TO_DotModels          writes external/protocols/NN-name/family.py and hashes
    python external/mealy_dot.py --same FOUND.pkl ...        (see same): is a machine found the model, exactly

No code of the collection is run: the .dot files are read as text.
"""
import hashlib, os, pickle, re, sys

MODELS = ["Bluetooth/CYBLE-416045-02.dot", "Bluetooth/bluetooth_model.dot", "Bluetooth/bluetooth_reduced.dot",
          "Bluetooth/cc2652r1.dot", "Bluetooth/CC2650.dot", "Bluetooth/nRF52832.dot",
          "Bluetooth/CC2640R2-no-pairing-req.dot", "Bluetooth/CC2640R2-no-feature-req.dot", "Bluetooth/CYW43455.dot",
          "TLS/miTLS_0.1.3_server_regular.dot", "TLS/OpenSSL_1.0.2_server_regular.dot",
          "TLS/NSS_3.17.4_server_regular.dot", "TLS/JSSE_1.8.0_25_server_regular.dot",
          "TLS/RSA_BSAFE_C_4.0.4_server_regular.dot",
          "MQTT/VerneMQ__two_client_will_retain.dot", "MQTT/hbmqtt__two_client_will_retain.dot",
          "MQTT/ActiveMQ__two_client_will_retain.dot", "MQTT/emqtt__two_client_will_retain.dot",
          "MQTT/mosquitto__two_client_will_retain.dot",
          "TCP/TCP_Linux_Client.dot", "TCP/tcp_server_windows_trans.dot", "TCP/tcp_server_bsd_trans.dot",
          "TCP/tcp_server_ubuntu_trans.dot"]


def read(path):
    """The inputs, the state begun in and the table (state, input) -> (state, output) of a Mealy machine in .dot."""
    text = open(path, encoding="utf-8", errors="replace").read()
    start, trans = None, {}
    for m in re.finditer(r'^\s*(\w+)\s*->\s*(\w+)\s*(?:\[(.*?)\])?;?\s*$', text, re.M):
        a, b, attrs = m.group(1), m.group(2), m.group(3) or ""
        if a.startswith("__start"):
            start = b
            continue
        lab = re.search(r'label\s*=\s*"(.*?)"', attrs) or re.search(r'label\s*=\s*<(.*)>', attrs)
        label = lab.group(1) if lab else ""
        i, o = re.split(r"\s*<br\s*/>\s*", label, maxsplit=1) if "<br" in label else label.split("/", 1)
        for one in i.split("|"):  # (an edge may name several inputs)
            trans[a, one.strip()] = (b, o.strip())
    states = sorted({a for a, _ in trans} | {b for b, _ in trans.values()})
    inputs = sorted({i for _, i in trans})
    start = start or ("s0" if "s0" in states else states[0])
    assert all((q, i) in trans for q in states for i in inputs), path  # (complete and deterministic)
    return inputs, start, trans


def reached(start, inputs, trans):
    seen, todo = {start}, [start]
    while todo:
        q = todo.pop()
        for i in inputs:
            if trans[q, i][0] not in seen:
                seen.add(trans[q, i][0])
                todo.append(trans[q, i][0])
    return seen


def same(machine, inputs, start, trans, most=500000):
    """Is the machine found the model, on every story? A walk in breadth over the pairs (state of the model,
    configuration of the machine): the shortest story they answer otherwise, or None if there is none (a machine with
    a whole number has no end of configurations: then None means none among the first `most` pairs)."""
    p0 = (start, machine.start, tuple(machine.zstart))
    seen, frontier = {p0}, [((), p0)]
    while frontier and len(seen) < most:
        nxt = []
        for w, (q, s, z) in frontier:
            for a in inputs:
                q2, out = trans[q, a]
                s2, z2 = machine.move(s, list(z), a)
                if machine.out(s2, z2) != out:
                    return w + (a,)
                p = (q2, s2, tuple(z2))
                if p not in seen:
                    seen.add(p)
                    nxt.append((w + (a,), p))
        frontier = nxt
    return None


def write(models, out):
    lines = []
    for n, rel in enumerate(MODELS, 1):
        inputs, start, trans = read(os.path.join(models, rel))
        name = f"{n:02d}-" + re.sub(r"[^a-z0-9]+", "-", os.path.basename(rel)[:-4].lower()).strip("-")
        os.makedirs(os.path.join(out, name), exist_ok=True)
        body = f'"""{rel}: a Mealy machine learned by others from a real implementation (AALpy, DotModels, MIT licence).\n' \
               f'{len(reached(start, inputs, trans))} states, {len(inputs)} inputs. The answer to a story of inputs is the ' \
               f'output to the last of them."""\n\n' \
               f"ALPHABET = {inputs!r}\nSTART = {start!r}\nTRANS = {{\n" + \
               "".join(f"    {k!r}: {v!r},\n" for k, v in sorted(trans.items())) + "}\n\n\n" \
               "def answer(seq):\n    q, out = START, \"\"\n    for a in seq:\n        q, out = TRANS[q, a]\n    return out\n"
        open(os.path.join(out, name, "family.py"), "w").write(body)
        lines.append(f"{hashlib.sha256(body.encode()).hexdigest()}  {name}/family.py")
    open(os.path.join(out, "hashes.txt"), "w").write("\n".join(lines) + "\n")
    print(len(lines), "families written to", out)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if sys.argv[1] == "--same":
        sys.path.insert(0, os.path.dirname(here))
        import importlib.util
        for path in sys.argv[2:]:  # FOLDER/NN-name.pkl, written by sealed_run.py --save FOLDER
            name = os.path.basename(path)[:-4]
            spec = importlib.util.spec_from_file_location("family", os.path.join(here, "protocols", name, "family.py"))
            fam = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fam)
            m = pickle.load(open(path, "rb"))
            w = None if m is None else same(m, fam.ALPHABET, fam.START, fam.TRANS)
            print(f"{name}: " + ("no machine" if m is None else "the model exactly" if w is None else
                                 f"not the model: they part on a story of {len(w)} inputs"))
    else:
        write(sys.argv[1], os.path.join(here, "protocols"))
