#!/usr/bin/env python3
"""
Training data for the Millennium digit demo: pictures of the digits 0-9 the way a player draws
them with blocks on a 20 x 20 canvas, normalized exactly like the JustMC code does it.

Sources
  * every TrueType / OpenType font found in the font folders (system + downloaded)
  * MNIST handwritten digits (idx files)
  * hand-made pixel glyphs 3x3, 3x4, 3x5, 5x7 drawn at scale 1..6
Each picture is "drawn" at a random block size (3..20 blocks high), optionally thinned to
1-block strokes (as players usually draw), then normalized:
  bounding box -> fit into N x N keeping the aspect ratio -> centered -> cell = OR of its source blocks

    python digits_data.py --fonts DIR [DIR ...] --mnist DIR --out data.npz
"""
import argparse
import glob
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

N = 10        # normalized picture: N x N, one row per time step
CANVAS = 20   # the canvas in the world is CANVAS x CANVAS blocks


# ----------------------------------------------------------------------------- normalization
def normalize(img):
    """img: 2-D 0/1 array (rows top->bottom, columns left->right), at most CANVAS x CANVAS.
    Returns the N x N 0/1 picture (uint8) or None when nothing is drawn.
    Integer arithmetic only, identical to dg_normalize in demo_digits.jc."""
    rows = np.where(img.any(1))[0]
    cols = np.where(img.any(0))[0]
    if len(rows) == 0:
        return None
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    s = max(h, w)
    oh = max(1, (2 * h * N + s) // (2 * s))
    ow = max(1, (2 * w * N + s) // (2 * s))
    oy = (N - oh) // 2
    ox = (N - ow) // 2
    out = np.zeros((N, N), np.uint8)
    for r in range(oh):
        a = (r * h) // oh
        b = ((r + 1) * h - 1) // oh
        for c in range(ow):
            p = (c * w) // ow
            q = ((c + 1) * w - 1) // ow
            if img[r0 + a:r0 + b + 1, c0 + p:c0 + q + 1].any():
                out[oy + r, ox + c] = 1
    return out


# ----------------------------------------------------------------------------- thinning
def thin(img):
    """Zhang-Suen thinning of a 0/1 picture -> 1-pixel strokes."""
    a = np.pad(img.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2 = a[:-2, 1:-1]; p3 = a[:-2, 2:]; p4 = a[1:-1, 2:]; p5 = a[2:, 2:]
            p6 = a[2:, 1:-1]; p7 = a[2:, :-2]; p8 = a[1:-1, :-2]; p9 = a[:-2, :-2]
            c = a[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            bsum = sum(x.astype(np.int32) for x in nb)
            seq = nb + [p2]
            trans = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (c == 1) & (bsum >= 2) & (bsum <= 6) & (trans == 1) & cond
            if rm.any():
                a[1:-1, 1:-1][rm] = 0
                changed = True
    return a[1:-1, 1:-1]


# ----------------------------------------------------------------------------- pixel glyphs
def _g(*rows):
    return np.array([[1 if ch == "#" else 0 for ch in r] for r in rows], np.uint8)


PIXEL = {
    # 3 x 3 (width x height): the smallest digits a player can draw
    "3x3": {
        0: [_g("###", "#.#", "###")],
        1: [_g("#", "#", "#"), _g("##.", ".#.", "###")],
        2: [_g("##.", ".#.", ".##")],
        3: [_g("###", ".##", "###")],
        4: [_g("#.#", "###", "..#")],
        5: [_g(".##", ".#.", "##.")],
        6: [_g("#..", "###", "###")],
        7: [_g("###", "..#", "..#")],
        8: [_g("###", "###", "###")],
        9: [_g("###", "###", "..#")],
    },
    # 3 x 4
    "3x4": {
        0: [_g("###", "#.#", "#.#", "###")],
        1: [_g("#", "#", "#", "#"), _g(".#.", "##.", ".#.", "###"), _g(".#", "##", ".#", ".#")],
        2: [_g("##.", "..#", ".#.", "###")],
        3: [_g("###", ".##", "..#", "###")],
        4: [_g("#.#", "###", "..#", "..#")],
        5: [_g("###", "##.", "..#", "##."), _g("###", "#..", "..#", "###")],
        6: [_g("#..", "###", "#.#", "###")],
        7: [_g("###", "..#", ".#.", ".#."), _g("###", "..#", "..#", "..#")],
        8: [_g("###", "###", "#.#", "###"), _g("###", "#.#", "###", "###")],
        9: [_g("###", "#.#", "###", "..#")],
    },
    # 3 x 5: the classic small pixel font
    "3x5": {
        0: [_g("###", "#.#", "#.#", "#.#", "###")],
        1: [_g("#", "#", "#", "#", "#"), _g(".#.", "##.", ".#.", ".#.", "###"), _g(".#", "##", ".#", ".#", ".#")],
        2: [_g("###", "..#", "###", "#..", "###"), _g("##.", "..#", ".#.", "#..", "###")],
        3: [_g("###", "..#", "###", "..#", "###"), _g("##.", "..#", ".#.", "..#", "##."), _g("###", "..#", ".##", "..#", "###")],
        4: [_g("#.#", "#.#", "###", "..#", "..#"), _g("..#", ".##", "#.#", "###", "..#")],
        5: [_g("###", "#..", "###", "..#", "###"), _g("###", "#..", "##.", "..#", "##.")],
        6: [_g("###", "#..", "###", "#.#", "###"), _g("#..", "#..", "###", "#.#", "###"), _g(".##", "#..", "###", "#.#", "###")],
        7: [_g("###", "..#", "..#", "..#", "..#"), _g("###", "..#", ".#.", ".#.", ".#."), _g("###", "..#", "..#", ".#.", ".#.")],
        8: [_g("###", "#.#", "###", "#.#", "###"), _g(".#.", "#.#", ".#.", "#.#", ".#.")],
        9: [_g("###", "#.#", "###", "..#", "###"), _g("###", "#.#", "###", "..#", "..#"), _g("###", "#.#", "###", "..#", "##.")],
    },
    # 5 x 7: the classic LCD font
    "5x7": {
        0: [_g(".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."), _g(".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###.")],
        1: [_g("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."), _g("..#", ".##", "#.#", "..#", "..#", "..#", "..#")],
        2: [_g(".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####")],
        3: [_g("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."), _g(".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###.")],
        4: [_g("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."), _g("#...#", "#...#", "#...#", "#####", "....#", "....#", "....#")],
        5: [_g("#####", "#....", "####.", "....#", "....#", "#...#", ".###.")],
        6: [_g("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###.")],
        7: [_g("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#...")],
        8: [_g(".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###.")],
        9: [_g(".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##..")],
    },
}


# ----------------------------------------------------------------------------- glyph sources
def list_fonts(dirs):
    files = []
    for d in dirs:
        for ext in ("ttf", "otf", "ttc", "TTF", "OTF"):
            files += glob.glob(os.path.join(d, "**", "*." + ext), recursive=True)
    out = []
    for f in sorted(set(files)):
        low = os.path.basename(f).lower()
        if any(k in low for k in ("emoji", "symbol", "dingbat", "webdings", "wingding", "music", "math")):
            continue
        out.append(f)
    return out


def render_font_digits(path, weight=None, size=96):
    """10 grayscale crops (float 0..1) of the digits of one font, or None if the font has no digits."""
    try:
        font = ImageFont.truetype(path, size)
        if weight is not None:
            try:
                font.set_variation_by_axes([weight])
            except Exception:
                return None
    except Exception:
        return None
    crops = []
    for d in range(10):
        im = Image.new("L", (size * 3, size * 3), 0)
        dr = ImageDraw.Draw(im)
        try:
            dr.text((size, size // 2), str(d), fill=255, font=font)
        except Exception:
            return None
        a = np.asarray(im, np.float32) / 255.0
        ys, xs = np.where(a > 0.2)
        if len(ys) == 0:
            return None
        crops.append(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    # a font without digits draws the same "missing glyph" box ten times
    sig = {(c.shape, round(float(c.sum()), 1)) for c in crops}
    if len(sig) < 8:
        return None
    return crops


def font_sources(paths):
    """[(name, [10 crops])] for every usable font (variable fonts in 3 weights)."""
    out = []
    for p in paths:
        weights = [None]
        try:
            f = ImageFont.truetype(p, 40)
            axes = f.get_variation_axes()
            if axes:
                lo, hi = axes[0]["minimum"], axes[0]["maximum"]
                weights = [lo + (hi - lo) * t for t in (0.2, 0.5, 0.9)]
        except Exception:
            pass
        for w in weights:
            crops = render_font_digits(p, w)
            if crops is not None:
                out.append((os.path.basename(p) + ("" if w is None else f"@{int(w)}"), crops))
    return out


def load_mnist(folder, part):
    pre = "train" if part == "train" else "t10k"
    with open(os.path.join(folder, f"{pre}-images-idx3-ubyte"), "rb") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with open(os.path.join(folder, f"{pre}-labels-idx1-ubyte"), "rb") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return x, y


# ----------------------------------------------------------------------------- "drawing" with blocks
def augment_hi(g, rng):
    """random rotation, shear, stretch and stroke weight of a high-resolution grayscale glyph"""
    im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
    pad = max(im.size) // 2 + 4
    big = Image.new("L", (im.size[0] + 2 * pad, im.size[1] + 2 * pad), 0)
    big.paste(im, (pad, pad))
    sh = rng.uniform(-0.28, 0.28)
    sx = rng.uniform(0.72, 1.3)
    w, h = big.size
    big = big.transform((w, h), Image.AFFINE, (1 / sx, sh, -sh * h / 2 + (w - w / sx) / 2, 0, 1, 0), Image.BILINEAR)
    big = big.rotate(rng.uniform(-12, 12), Image.BILINEAR)
    k = rng.choice([0, 0, 0, 0, 3, 5, -3])
    if k > 0:
        big = big.filter(ImageFilter.MaxFilter(k))
    elif k < 0:
        big = big.filter(ImageFilter.MinFilter(-k))
    a = np.asarray(big, np.float32) / 255.0
    ys, xs = np.where(a > 0.2)
    if len(ys) == 0:
        return g
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def to_blocks(g, hb, rng):
    """area-resize a grayscale glyph to hb blocks high (width by its aspect), threshold -> 0/1"""
    h, w = g.shape
    wb = int(round(hb * w / h))
    wb = max(1, min(CANVAS, wb))
    hb = max(1, min(CANVAS, hb))
    im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).resize((wb, hb), Image.BOX)
    a = np.asarray(im, np.float32) / 255.0
    thr = rng.uniform(0.15, 0.45)
    b = (a >= thr).astype(np.uint8)
    if b.sum() == 0:
        b = (a >= a.max() * 0.5).astype(np.uint8)
    return b


def sloppy(b, rng):
    """a player's small mistakes: one extra block next to a stroke, or one missing block"""
    b = b.copy()
    r = rng.random()
    if r < 0.07:
        on = np.argwhere(b == 1)
        if len(on) > 4:
            y, x = on[rng.randrange(len(on))]
            b[y, x] = 0
    elif r < 0.17:
        on = np.argwhere(b == 1)
        y, x = on[rng.randrange(len(on))]
        dy, dx = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        y2, x2 = y + dy, x + dx
        if 0 <= y2 < b.shape[0] and 0 <= x2 < b.shape[1]:
            b[y2, x2] = 1
    return b


def plausible(x):
    """drop degenerate drawings: holes closed into a solid block, or almost nothing left"""
    if x is None:
        return None
    f = x.mean()
    return x if 0.06 <= f <= 0.6 else None


def draw_glyph(g, rng, hmin, hmax, thin_p=0.55):
    """one block drawing of a grayscale glyph -> normalized N x N picture (or None)"""
    g = augment_hi(g, rng)
    hb = rng.randint(hmin, hmax)
    b = to_blocks(g, hb, rng)
    if hb >= 6 and rng.random() < thin_p:
        t = thin(b)
        if t.sum() >= 3:
            b = t
    b = sloppy(b, rng)
    return plausible(normalize(b))


def draw_pixel(p, rng):
    """a pixel glyph drawn at a random integer scale (thick strokes or thinned)"""
    h, w = p.shape
    ky = rng.randint(1, max(1, CANVAS // h))
    kx = rng.randint(max(1, ky - 1), max(1, min(CANVAS // w, ky + 1)))
    b = np.kron(p, np.ones((ky, kx), np.uint8))
    if b.shape[0] >= 8 and rng.random() < 0.5:
        b = sloppy(b, rng)
    return normalize(b)


# ----------------------------------------------------------------------------- dataset
def _work(job):
    kind, seed, payload = job
    rng = random.Random(seed)
    xs, ys, src = [], [], []
    if kind == "font":
        name, crops, reps = payload
        for _ in range(reps):
            for d in range(10):
                x = draw_glyph(crops[d], rng, 5, 20)
                if x is not None:
                    xs.append(x); ys.append(d); src.append(0)
    elif kind == "mnist":
        imgs, labels = payload
        for img, d in zip(imgs, labels):
            g = img.astype(np.float32) / 255.0
            g = np.kron(g, np.ones((3, 3), np.float32))
            ys_, xs_ = np.where(g > 0.2)
            g = g[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
            x = draw_glyph(g, rng, 6, 20, thin_p=0.6)
            if x is not None:
                xs.append(x); ys.append(int(d)); src.append(1)
    else:
        reps = payload
        for _ in range(reps):
            for size, table in PIXEL.items():
                for d, variants in table.items():
                    x = draw_pixel(variants[rng.randrange(len(variants))], rng)
                    if x is not None:
                        xs.append(x); ys.append(d); src.append(2)
    return np.array(xs, np.uint8), np.array(ys, np.uint8), np.array(src, np.uint8)


def build(fonts, mnist_x, mnist_y, font_reps, mnist_reps, pixel_reps, seed):
    jobs = []
    for i, (name, crops) in enumerate(fonts):
        jobs.append(("font", seed * 1000003 + i, (name, crops, font_reps)))
    step = 2000
    for r in range(mnist_reps):
        for i in range(0, len(mnist_x), step):
            jobs.append(("mnist", seed * 7919 + r * 100003 + i, (mnist_x[i:i + step], mnist_y[i:i + step])))
    for i in range(max(1, pixel_reps // 50)):
        jobs.append(("pixel", seed * 104729 + i, min(50, pixel_reps)))
    with Pool(os.cpu_count()) as pool:
        parts = pool.map(_work, jobs, chunksize=1)
    parts = [p for p in parts if len(p[0])]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    s = np.concatenate([p[2] for p in parts])
    return x, y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", nargs="+", required=True)
    ap.add_argument("--mnist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--font-reps", type=int, default=60)
    ap.add_argument("--mnist-reps", type=int, default=2)
    ap.add_argument("--pixel-reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    fonts = font_sources(list_fonts(a.fonts))
    rng = random.Random(a.seed)
    rng.shuffle(fonts)
    n_val = max(3, len(fonts) // 10)
    val_fonts, train_fonts = fonts[:n_val], fonts[n_val:]
    print(f"fonts: {len(fonts)} faces ({len(train_fonts)} train, {len(val_fonts)} validation)")
    print("validation fonts:", ", ".join(n for n, _ in val_fonts))

    mx, my = load_mnist(a.mnist, "train")
    tx, ty = load_mnist(a.mnist, "test")
    xtr, ytr, str_ = build(train_fonts, mx, my, a.font_reps, a.mnist_reps, a.pixel_reps, a.seed)
    xva, yva, sva = build(val_fonts, tx, ty, max(10, a.font_reps // 4), 1, max(50, a.pixel_reps // 4), a.seed + 1)
    print(f"train {len(xtr)} pictures, validation {len(xva)}")
    for name, arr, src in (("train", ytr, str_), ("val", yva, sva)):
        print(name, "per digit:", np.bincount(arr, minlength=10).tolist(), "per source (font, mnist, pixel):", np.bincount(src, minlength=3).tolist())
    np.savez_compressed(a.out, xtr=xtr, ytr=ytr, str=str_, xva=xva, yva=yva, sva=sva)


if __name__ == "__main__":
    main()
