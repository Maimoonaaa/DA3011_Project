
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import softdtw_div as sd

DEFAULT_DATASETS = ["GunPoint", "ECG200", "CBF", "ItalyPowerDemand", "Coffee", "TwoLeadECG",
                    "ArrowHead", "Trace", "Plane", "MoteStrain", "SonyAIBORobotSurface1",
                    "SyntheticControl"]
METHODS = ["euclid", "dtw", "sdtw", "sdtw_div", "sharp", "sharp_div"]
LABELS = {"euclid": "Euclidean", "dtw": "DTW", "sdtw": "SDTW", "sdtw_div": "SDTW div",
          "sharp": "Sharp", "sharp_div": "Sharp div"}


#  Data loading
def _znorm(X):
    mu = X.mean(1, keepdims=True)
    sg = X.std(1, keepdims=True)
    sg[sg == 0] = 1.0
    return (X - mu) / sg


def make_synthetic(n_train=30, n_test=90, T=100, seed=0):
    """Tiny offline dataset: 3 classes of randomly time-warped shapes + noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)

    def gen(n):
        X, y = [], []
        for i in range(n):
            c = i % 3
            warp = np.clip(t + 0.08 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t + rng.uniform(0, 6)), 0, 1)
            if c == 0:
                s = np.sin(2 * np.pi * 2 * warp)
            elif c == 1:
                s = np.exp(-((warp - 0.5) ** 2) / 0.01)
            else:
                s = (warp > 0.4).astype(float) * (warp < 0.7)
            X.append(s + 0.15 * rng.normal(size=T))
            y.append(c)
        return np.array(X)[:, :, None], np.array(y)

    Xtr, ytr = gen(n_train)
    Xte, yte = gen(n_test)
    return Xtr, ytr, Xte, yte


def load_ucr(name, data_dir="data/UCR"):
    """Returns Xtr (N,T,1), ytr, Xte, yte (labels encoded 0..K-1), z-normalised.
    Order tried: local UCR-2018 .tsv files -> tslearn downloader -> aeon downloader."""
    if name.lower() == "synthetic":
        Xtr, ytr, Xte, yte = make_synthetic()
    else:
        ptr = os.path.join(data_dir, name, f"{name}_TRAIN.tsv")
        pte = os.path.join(data_dir, name, f"{name}_TEST.tsv")
        Xtr = None
        if os.path.exists(ptr) and os.path.exists(pte):
            tr, te = np.loadtxt(ptr, delimiter="\t"), np.loadtxt(pte, delimiter="\t")
            ytr, Xtr, yte, Xte = tr[:, 0], tr[:, 1:, None], te[:, 0], te[:, 1:, None]
        if Xtr is None:
            try:
                from tslearn.datasets import UCR_UEA_datasets
                Xtr, ytr, Xte, yte = UCR_UEA_datasets().load_dataset(name)
            except Exception as e:  # noqa
                try:
                    from aeon.datasets import load_classification
                    Xtr, ytr = load_classification(name, split="train")
                    Xte, yte = load_classification(name, split="test")
                    Xtr, Xte = np.transpose(Xtr, (0, 2, 1)), np.transpose(Xte, (0, 2, 1))
                except Exception as e2:  # noqa
                    raise RuntimeError(
                        f"Could not load '{name}'. Download the UCR archive manually and unzip it to "
                        f"{data_dir}/ (see README). tslearn error: {e}; aeon error: {e2}")
    Xtr, Xte = np.asarray(Xtr, dtype=np.float64), np.asarray(Xte, dtype=np.float64)
    if np.isnan(Xtr).any() or np.isnan(Xte).any():
        raise ValueError(f"{name} has variable-length (NaN padded) series - skipped (needs equal length).")
    classes = np.unique(np.concatenate([ytr, yte]).astype(str))
    cmap = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cmap[str(v)] for v in ytr])
    yte = np.array([cmap[str(v)] for v in yte])
    return (np.ascontiguousarray(_znorm(Xtr)), ytr, np.ascontiguousarray(_znorm(Xte)), yte)


#  sanity: numerical verification of the paper's claims
def cmd_sanity(args):
    rng = np.random.default_rng(0)
    print("\n[1] Gradient check (analytic vs central finite differences, gamma=1)")
    X, Y = rng.normal(size=(6, 2)), rng.normal(size=(8, 2))
    for kind in sd.KINDS[1:]:
        v, g = sd.value_and_grad(X, Y, kind, 1.0)
        gn = np.zeros_like(X)
        for i in range(X.shape[0]):
            for k in range(X.shape[1]):
                Xp, Xm = X.copy(), X.copy()
                Xp[i, k] += 1e-6
                Xm[i, k] -= 1e-6
                gn[i, k] = (sd.value_and_grad(Xp, Y, kind, 1.0)[0] - sd.value_and_grad(Xm, Y, kind, 1.0)[0]) / 2e-6
        print(f"   {kind:10s} value={v:9.4f}  rel. grad error={np.abs(g - gn).max() / np.abs(gn).max():.2e}")

    print("\n[2] Soft-DTW limitations (Prop. 2): sdtw(C(X,X)) <= 0 and can be negative; divergence is 0")
    for gamma in (0.1, 1.0, 10.0):
        vals = []
        for _ in range(100):
            Z = rng.normal(size=(rng.integers(5, 20), 1))
            vals.append(sd.value_and_grad(Z, Z, "sdtw", gamma)[0])
        d = [sd.value_and_grad(Z, Z, "sdtw_div", gamma)[0], sd.value_and_grad(Z, Z, "sharp_div", gamma)[0]]
        print(f"   gamma={gamma:5.1f}: max sdtw(X,X)={max(vals):+.4f}  (always <= 0)   "
              f"D(X,X)={d[0]:.1e}  S(X,X)={d[1]:.1e}")

    print("\n[3] Non-negativity of the divergences on 300 random pairs (different lengths allowed, d=1)")
    for kind in ("sdtw_div", "sharp_div"):
        for gamma in (0.1, 1.0, 10.0):
            mn = np.inf
            for _ in range(300):
                A = rng.normal(size=(rng.integers(5, 15), 1))
                B = rng.normal(size=(rng.integers(5, 15), 1))
                mn = min(mn, sd.value_and_grad(A, B, kind, gamma)[0])
            print(f"   {kind:10s} gamma={gamma:5.1f}  min over pairs = {mn:+.4f}")
        # soft-DTW itself
    neg = 0
    for _ in range(300):
        A = rng.normal(size=(10, 1))
        B = rng.normal(size=(10, 1))
        neg += sd.value_and_grad(A, B, "sdtw", 10.0)[0] < 0
    print(f"   (for comparison) plain soft-DTW is negative in {neg}/300 pairs at gamma=10")

    print("\n[4] Stationarity at X=Y (Prop. 4): ||grad_X||")
    Y = rng.normal(size=(15, 1))
    for kind in ("sdtw", "sharp", "sdtw_div", "sharp_div"):
        print(f"   {kind:10s} {np.linalg.norm(sd.value_and_grad(Y, Y, kind, 1.0)[1]):.2e}")
    print("   -> biased versions are NOT stationary at X=Y (not minimised there); divergences are.")

#  demo: Fig 2 (denoising), Fig 3 (barycenters), Fig 4 (interpolation)
def cmd_demo(args):
    os.makedirs(args.out, exist_ok=True)
    Xtr, ytr, _, _ = load_ucr(args.dataset, args.data_dir)
    rng = np.random.default_rng(args.seed)

    #Fig. 2: argmin_X sdtw(C(X,Y)) with X=Y init moves away from Y; the divergence stays put
    Y = Xtr[0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    for a, (kind, g) in zip(ax, [("sdtw", 0.1), ("sdtw", 10.0), ("sdtw_div", 10.0)]):
        X = sd.barycenter(Y[None], kind, g, init=Y, maxiter=200)
        a.plot(Y[:, 0], color="lightgray", lw=4, label="target Y")
        a.plot(X[:, 0], lw=2, label=f"argmin {LABELS[kind]}")
        a.set_title(f"{LABELS[kind]}, gamma={g}  (||X-Y||={np.linalg.norm(X - Y):.2f})")
        a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig2_denoising_{args.dataset}.png", dpi=130)
    plt.close()

    #Fig. 3: barycenters of 10 series of one class
    idx = np.where(ytr == ytr[0])[0]
    idx = rng.permutation(idx)[:10]
    Ys = Xtr[idx]
    fig, axes = plt.subplots(2, 4, figsize=(15, 5), sharey=True)
    order = ["euclid", "sdtw", "sharp", "dtw", "sdtw_div", "sharp_div"]
    for a, kind in zip(axes.ravel(), order):
        t0 = time.time()
        B = sd.barycenter(Ys, kind, args.gamma, maxiter=args.iters)
        for y in Ys:
            a.plot(y[:, 0], color="gray", alpha=0.25)
        a.plot(B[:, 0], lw=2.5)
        a.set_title(LABELS[kind])
        print(f"   barycenter {kind:10s} {time.time() - t0:5.1f}s")
    for a in axes.ravel()[len(order):]:
        a.axis("off")
    plt.suptitle(f"Averages of {len(Ys)} series ({args.dataset}), gamma={args.gamma}")
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig3_barycenters_{args.dataset}.png", dpi=130)
    plt.close()

    #Fig. 4: interpolation  w1=pi, w2=1-pi
    Y1, Y2 = Xtr[idx[0]], Xtr[idx[1]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharey=True)
    for a, kind in zip(axes.ravel(), ["euclid", "dtw", "sdtw", "sharp", "sdtw_div", "sharp_div"]):
        for pi, col in zip([0.0, 0.25, 0.5, 0.75, 1.0], plt.cm.viridis(np.linspace(0.9, 0.1, 5))):
            if pi in (0.0, 1.0):
                B = Y2 if pi == 0.0 else Y1
            else:
                B = sd.barycenter(np.stack([Y1, Y2]), kind, args.gamma, weights=[pi, 1 - pi], maxiter=args.iters)
            a.plot(B[:, 0], color=col)
        a.set_title(LABELS[kind])
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig4_interpolation_{args.dataset}.png", dpi=130)
    plt.close()
    print(f"Figures saved in {args.out}/")


#  classify: 1-NN and nearest-centroid  (Tables 2/3 of the paper, smaller scale)
def run_dataset(name, args):
    Xtr, ytr, Xte, yte = load_ucr(name, args.data_dir)
    rng = np.random.default_rng(args.seed)
    if args.max_train and len(Xtr) > args.max_train:
        keep = np.sort(rng.permutation(len(Xtr))[:args.max_train])
        Xtr, ytr = Xtr[keep], ytr[keep]
    if args.max_test and len(Xte) > args.max_test:
        keep = np.sort(rng.permutation(len(Xte))[:args.max_test])
        Xte, yte = Xte[keep], yte[keep]
    n = len(Xtr)
    print(f"\n=== {name}: train={n} test={len(Xte)} length={Xtr.shape[1]} classes={len(np.unique(ytr))}")

    #cached raw (uncorrected) matrices, shared by biased and corrected variants
    cache = {}

    def tr(base, g):
        if ("tr", base, g) not in cache:
            cache[("tr", base, g)] = (sd.raw_matrix(Xtr, Xtr, base, g), sd.self_terms(Xtr, base, g))
        return cache[("tr", base, g)]

    def te(base, g):
        if ("te", base, g) not in cache:
            cache[("te", base, g)] = (sd.raw_matrix(Xte, Xtr, base, g), sd.self_terms(Xte, base, g),
                                      tr(base, g)[1])
        return cache[("te", base, g)]

    def correct(R, sa, sb, kind):
        return R - 0.5 * sa[:, None] - 0.5 * sb[None, :] if kind.endswith("_div") else R

    #gamma selection by repeated 2/3 - 1/3 hold-out (as in the paper)
    n_fit = int(round(2 * n / 3))
    splits = [rng.permutation(n) for _ in range(args.cv_repeats)]
    gam = {"euclid": None, "dtw": None}
    for kind in ("sdtw", "sdtw_div", "sharp", "sharp_div"):
        base = sd.base_of(kind)
        best_acc, best_g = -1, None
        for g in args.gammas:
            R, s = tr(base, g)
            D = correct(R, s, s, kind)
            accs = []
            for perm in splits:
                a, b = perm[:n_fit], perm[n_fit:]
                pred = ytr[a][np.argmin(D[np.ix_(b, a)], axis=1)]
                accs.append(np.mean(pred == ytr[b]))
            if np.mean(accs) > best_acc:
                best_acc, best_g = np.mean(accs), g
        gam[kind] = best_g

    rows = []
    #1-NN
    acc_nn = {}
    for kind in METHODS:
        t0 = time.time()
        g = gam[kind] or 1.0
        base = sd.base_of(kind)
        R, sa, sb = te(base, g)
        D = correct(R, sa, sb, kind)
        acc_nn[kind] = float(np.mean(ytr[np.argmin(D, axis=1)] == yte))
        print(f"  1-NN  {LABELS[kind]:10s} acc={acc_nn[kind]:.4f}  gamma={gam[kind]}  ({time.time() - t0:.1f}s)")
    rows.append({"dataset": name, "task": "1nn", **acc_nn, **{f"gamma_{k}": v for k, v in gam.items()}})

    #nearest centroid
    if args.centroid:
        classes = np.unique(ytr)
        acc_nc = {}
        for kind in METHODS:
            t0 = time.time()
            g = gam[kind] or 1.0
            cents = []
            for c in classes:
                Yc = Xtr[ytr == c]
                if len(Yc) > args.max_per_class:
                    Yc = Yc[rng.permutation(len(Yc))[:args.max_per_class]]
                cents.append(sd.barycenter(Yc, kind, g, maxiter=args.iters))
            cents = np.stack(cents)
            D = sd.discrepancy_matrix(Xte, cents, kind, g)
            acc_nc[kind] = float(np.mean(classes[np.argmin(D, axis=1)] == yte))
            print(f"  NC    {LABELS[kind]:10s} acc={acc_nc[kind]:.4f}  ({time.time() - t0:.1f}s)")
        rows.append({"dataset": name, "task": "centroid", **acc_nc, **{f"gamma_{k}": v for k, v in gam.items()}})
    return rows


def win_table(df):
    """Paper's Table 2/3: % of datasets where method A (row) is within 99% or better than B (col)."""
    T = pd.DataFrame(index=[LABELS[m] for m in METHODS], columns=[LABELS[m] for m in METHODS], dtype=float)
    for a in METHODS:
        for b in METHODS:
            T.loc[LABELS[a], LABELS[b]] = np.nan if a == b else 100 * np.mean(df[a].values >= 0.99 * df[b].values)
    return T.round(1)


def cmd_classify(args):
    os.makedirs(args.out, exist_ok=True)
    all_rows = []
    for name in args.datasets:
        try:
            all_rows += run_dataset(name, args)
        except Exception as e:  # noqa
            print(f"  !! skipping {name}: {e}")
    if not all_rows:
        return
    res = pd.DataFrame(all_rows)
    res.to_csv(f"{args.out}/classification_results.csv", index=False)
    for task, title in (("1nn", "Nearest neighbour"), ("centroid", "Nearest centroid")):
        df = res[res.task == task]
        if df.empty:
            continue
        print(f"\n##### {title}: accuracy per dataset")
        print((df.set_index("dataset")[METHODS] * 100).round(2).rename(columns=LABELS).to_string())
        print(f"\nmean accuracy (%): " + ", ".join(f"{LABELS[m]}={100 * df[m].mean():.2f}" for m in METHODS))
        print(f"mean rank (1=best):  " + ", ".join(
            f"{LABELS[m]}={df[METHODS].rank(axis=1, ascending=False)[m].mean():.2f}" for m in METHODS))
        W = win_table(df)
        print(f"\n{title}: % of datasets where row is within 99% or better than column (Table 2/3 style)")
        print(W.to_string())
        W.to_csv(f"{args.out}/win_table_{task}.csv")
        ax = (df.set_index("dataset")[METHODS] * 100).rename(columns=LABELS).plot.bar(figsize=(12, 4))
        ax.set_ylabel("accuracy (%)")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f"{args.out}/accuracy_{task}.png", dpi=130)
        plt.close()
    print(f"\nSaved results to {args.out}/")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sanity", cmd_sanity), ("demo", cmd_demo), ("classify", cmd_classify)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--data-dir", default="data/UCR")
        s.add_argument("--out", default="results")
        s.add_argument("--seed", type=int, default=0)
        if name == "demo":
            s.add_argument("--dataset", default="GunPoint")
            s.add_argument("--gamma", type=float, default=10.0)
            s.add_argument("--iters", type=int, default=100)
        if name == "classify":
            s.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
            s.add_argument("--gammas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
            s.add_argument("--cv-repeats", type=int, default=3)
            s.add_argument("--centroid", action="store_true", help="also run nearest-centroid (slower)")
            s.add_argument("--max-per-class", type=int, default=20, help="series per class for barycenters")
            s.add_argument("--iters", type=int, default=50, help="L-BFGS iterations for barycenters")
            s.add_argument("--max-train", type=int, default=0, help="subsample train set (0 = all)")
            s.add_argument("--max-test", type=int, default=0, help="subsample test set (0 = all)")
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
