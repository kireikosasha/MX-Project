#!/usr/bin/env python3
"""
The whole chat model in one command:

    python justmc/tools/chat/run_all.py --work ../millennium-chat                 (the 6 M model)
    python justmc/tools/chat/run_all.py --work ../millennium-chat --model 13m     (the big one: LSTM 512,
        embedding 512, numbers as slots and step-by-step calculation, three exchanges; -> weights/chat13m)

  1. download.py        datasets -> <work>/raw
  2. chat_sources.py    dialogues -> <work>/v2/src
  3. chat_translate.py  more Russian / Ukrainian dialogues by machine translation (skipped if it cannot run)
  4. chat_data.py       training sequences -> <work>/v2/data, fine-tuning set -> <work>/v2/data_ft
  5. chat_train.py      the model (~6 M parameters) -> <work>/v2/out, fine-tuned on the bot's own
                        dialogues -> <work>/v2/out_ft
  6. chat_export.py     JustMC files -> justmc/weights/chat
  7. git add / commit / push of justmc/weights/chat (skip with --no-push); then in the world: ~ai install

A step whose result is already there is skipped, an interrupted training goes on from its last epoch, so the
command can simply be run again. With a CUDA GPU PyTorch uses it (--device cpu to force the processor).
Smaller and faster model: --hidden 192 (every LSTM neuron costs JustMC ~50 actions per token).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(REPO, ".."))
j = os.path.join
# hidden / dim / attention sizes, vocabulary, data builder, folder under <work>, folder under weights/, training flags
PRESETS = {
    "6m": {"hidden": 256, "dim": 256, "da": 64, "dv": 128, "vocab": "vocab.json", "builder": "chat_data.py", "sub": "v2",
           "name": "chat", "batch": 384, "flags": [], "temp": "0.5"},
    "13m": {"hidden": 512, "dim": 512, "da": 128, "dv": 256, "vocab": "vocab_big.json", "builder": "chat_data_big.py", "sub": "big",
            "name": "chat13m", "batch": 384, "flags": [], "temp": "0.3"},
}


def run(args, ok=(0,)):
    print(">>", " ".join(os.path.relpath(x) if os.path.isabs(x) and os.path.exists(x) else x for x in args), flush=True)
    r = subprocess.run([sys.executable, "-u"] + args)
    if r.returncode not in ok:
        sys.exit(r.returncode)
    return r.returncode


def ensure_translation_packages():
    """transformers 4.50+ loads the Opus-MT weights only with PyTorch 2.6+; 4.46.3 works with older PyTorch too"""
    try:
        import torch
        tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:  # noqa: BLE001
        sys.exit("PyTorch is not installed: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    try:
        import sentencepiece  # noqa: F401
        import sacremoses  # noqa: F401
        import transformers
        fv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        good = tv >= (2, 6) or fv < (4, 50)
    except Exception:  # noqa: BLE001
        good = False
    if not good:
        print(">> installing transformers 4.46.3 for the translation step (works with this PyTorch)", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.46.3", "sentencepiece", "sacremoses"])


def trained(path):
    if not os.path.exists(path):
        return False
    import torch
    return bool(torch.load(path, map_location="cpu", weights_only=False).get("done"))


def git(*args):
    return subprocess.run(["git", "-C", ROOT] + list(args), capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="folder for datasets, translations, checkpoints (several GB)")
    ap.add_argument("--model", default="6m", choices=sorted(PRESETS), help="which model to train")
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--dim", type=int)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--ft-epochs", type=int, default=2)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="", help="suffix of the checkpoint folders (several models side by side)")
    ap.add_argument("--ru-mb", type=int, default=200)
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="train on N random sequences only (a quick trial run)")
    ap.add_argument("--no-pip", action="store_true", help="do not install anything")
    ap.add_argument("--no-push", action="store_true", help="do not commit and push the exported model")
    ap.add_argument("--out")
    a = ap.parse_args()
    P = PRESETS[a.model]
    VOCAB = j(HERE, P["vocab"])
    a.out = a.out or j(REPO, "weights", P["name"])
    a.hidden, a.dim, a.batch = a.hidden or P["hidden"], a.dim or P["dim"], a.batch or P["batch"]
    w = os.path.abspath(a.work)
    src = j(w, "v2", "src")                     # the dialogues and their translations are shared by the models
    v2 = j(w, P["sub"])
    dev = ["--device", a.device] if a.device else []
    run([j(HERE, "download.py"), "--out", j(w, "raw"), "--ru-mb", str(a.ru_mb)])
    if not (os.path.exists(j(src, "dialogs.jsonl")) and os.path.exists(j(src, "pairs.jsonl"))):
        run([j(HERE, "chat_sources.py"), "--raw", j(w, "raw"), "--out", src])
    if not a.no_translate and not os.path.exists(j(src, "pairs_uk.jsonl")):
        if not a.no_pip:
            ensure_translation_packages()
        if run([j(HERE, "chat_translate.py"), "--src", src, "--cache", j(w, "v2", "mt")] + dev, ok=(0, 3)) == 3:
            print(">> going on without the translated dialogues", flush=True)
    # the data is rebuilt when the translations appear later
    mark = "mt" if os.path.exists(j(src, "pairs_uk.jsonl")) else "nomt"
    common = ["--src", src, "--qa", j(w, "raw"), "--vocab", VOCAB]
    for name, extra in (("data", []), ("data_ft", ["--ft"])):
        d = j(v2, name)
        if not (os.path.exists(j(d, "chat_data.npz")) and os.path.exists(j(d, mark))):
            run([j(HERE, P["builder"])] + common + ["--out", d] + extra)
            open(j(d, mark), "w").close()
    out, out_ft = j(v2, "out" + a.tag), j(v2, "out_ft" + a.tag)
    size = ["--hidden", str(a.hidden), "--dim", str(a.dim), "--da", str(P["da"]), "--dv", str(P["dv"]),
            "--batch", str(a.batch)] + P["flags"]
    if a.limit:
        size += ["--limit", str(a.limit)]
    if not trained(j(out, "chat.pt")):
        run([j(HERE, "chat_train.py"), "--data", j(v2, "data"), "--out", out, "--epochs", str(a.epochs), "--resume"] + size + dev)
    if not trained(j(out_ft, "chat.pt")):
        run([j(HERE, "chat_train.py"), "--data", j(v2, "data_ft"), "--out", out_ft, "--init", j(out, "chat.pt"),
             "--epochs", str(a.ft_epochs), "--lr", "8e-4", "--resume"] + size + dev)
    export = [j(HERE, "chat_export.py"), "--model", j(out_ft, "chat.pt"), "--vocab", VOCAB, "--out", a.out, "--name", P["name"],
              "--temp", P["temp"]]
    if a.model != "6m":                         # decision thresholds chosen on the validation set
        run([j(HERE, "chat_thresholds.py"), "--model", j(out_ft, "chat.pt"), "--data", j(v2, "data"),
             "--out", j(out_ft, "thresholds.json")] + dev)
        export += ["--thresholds", j(out_ft, "thresholds.json")]
    run(export)
    rel = os.path.relpath(a.out, ROOT).replace("\\", "/")
    if a.no_push:
        print(f"done: {a.out}\nnow: git add {rel} && git commit -m \"chat model\" && git push, then ~ai install in the world")
        return
    git("add", "-A", rel)
    c = git("commit", "-m", "JustMC chat: trained model" + ("" if a.model == "6m" else f" ({a.model})"), "--", rel)
    git("pull", "--rebase", "--autostash")
    p = git("push")
    if p.returncode == 0:
        print("done: the model is pushed. In the world: reload the module if it is old, then ~ai install")
    else:
        print((c.stdout + c.stderr + p.stdout + p.stderr).strip()[-800:])
        print(f"the model is in {a.out}, but git could not push it. Run yourself:\n"
              f"  git add {rel}\n  git commit -m \"chat model\"\n  git push\nthen ~ai install in the world")


if __name__ == "__main__":
    main()
