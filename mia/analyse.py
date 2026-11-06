#!/usr/bin/env python3
import argparse, glob, json, multiprocessing, os, sys, zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

K_FRAC, FPR, GAMMA, SEED = 0.2, 0.01, 2.0, 2
A_GRID = np.round(np.arange(0, 1.01, 0.1), 1)


def midranks(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    start = np.r_[True, xs[1:] != xs[:-1]]
    first = np.flatnonzero(start)
    last = np.r_[first[1:], len(x)] - 1
    run = np.cumsum(start) - 1
    r = np.empty(len(x))
    r[order] = (first[run] + last[run]) / 2.0 + 1.0
    return r


def auc(s, l):
    keep = ~np.isnan(s)   # NaN: not scored (LiRA without an OUT sibling)
    s, l = s[keep], l[keep]
    n1 = l.sum()
    return float((midranks(s)[l].sum() - n1 * (n1 + 1) / 2.0) / (n1 * (len(l) - n1)))


def tpr_at_fpr(s, l):
    """A record is called a member iff its score exceeds the threshold, so ties at the threshold count against."""
    keep = ~np.isnan(s)
    s, l = s[keep], l[keep]
    neg = np.sort(s[~l])[::-1]
    allowed = int(np.floor(FPR * len(neg)))
    thr = neg[allowed] if allowed < len(neg) else -np.inf
    fp = int((s[~l] > thr).sum())
    return {"tpr": float((s[l] > thr).mean()), "fp": fp, "n0": len(neg), "achieved_fpr": fp / len(neg)}


def resamples(groups, B, rng):
    """Cluster bootstrap: content groups drawn with replacement, each with all its records."""
    _, inv = np.unique(groups, return_inverse=True)
    parts = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    return [np.concatenate([parts[g] for g in rng.integers(0, len(parts), len(parts))]) for _ in range(B)]


def ci(draws):
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


class Scores:
    def __init__(self, path):
        z = np.load(path)
        self.path, self.h = path, json.loads(str(z["header"]))
        for k in z.files:
            if k != "header":
                setattr(self, k, z[k])

    def key(self):
        return self.h["task"], self.h["model_name"], self.h["record_seed"]

    def idx(self, role):
        sel = np.where(self.role == role)[0]
        return sel[np.argsort(self.k[sel])]

    def span(self, i):
        o, n = self.tok_off[i], self.n_tok[i]
        return slice(o + n - self.opt_len[i], o + n)

    def full(self, i):
        return slice(self.tok_off[i], self.tok_off[i] + self.n_tok[i])

    def prob(self):
        if self.h["task"] == "SST2":
            cl = self.cand_lp
            return np.exp(cl[np.arange(len(cl)), self.y] - np.logaddexp(cl[:, 0], cl[:, 1]))
        return np.exp(-self.loss)

    def phi(self):
        if self.h["task"] == "SST2":
            cl = self.cand_lp
            return cl[np.arange(len(cl)), self.y] - cl[np.arange(len(cl)), 1 - self.y]
        out = np.empty(len(self.loss))
        for i in range(len(out)):
            p = np.clip(np.exp(self.tok_lp[self.span(i)].astype(np.float64)), 1e-12, 1 - 1e-12)
            out[i] = np.mean(np.log(p) - np.log1p(-p))
        return out

    # Min-K%, Detecting Pretraining Data from Large Language Models, https://arxiv.org/abs/2310.16789. Min-K%++,
    # Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models,
    # https://arxiv.org/abs/2404.02936. At least one token is kept, since 20% of a span under five tokens rounds to zero.
    def mink(self, plus):
        out = np.empty(len(self.loss))
        for i in range(len(out)):
            sl = self.full(i)
            v = self.tok_lp[sl].astype(np.float64)
            if plus:
                v = (v - self.tok_mu[sl]) / np.maximum(self.tok_sd[sl].astype(np.float64), 1e-8)
            out[i] = np.sort(v)[:max(1, int(K_FRAC * len(v)))].mean()
        return out

    # zlib ratio, Extracting Training Data from Large Language Models, https://arxiv.org/abs/2012.07805
    def zlib(self):
        nll = np.array([-self.tok_lp[self.full(i)].astype(np.float64).mean() for i in range(len(self.loss))])
        return -nll / self.zlib_len

    # Gap attack, Label-Only Membership Inference Attacks, https://arxiv.org/abs/2007.14321. SQuAD uses the greedy-decode F1.
    def label(self):
        return (self.pred == self.y).astype(float) if self.h["task"] == "SST2" else self.dec_f1


# EZ-MIA, Powerful Training-Free Membership Inference Against Fine-Tuned Autoregressive Language Models,
# https://arxiv.org/abs/2601.12104. Edge cases follow the authors' code: the first scored position is dropped, and a
# record with fewer than two error positions or with N = 0 scores 0.
def ez(x, base):
    out = np.empty(len(x.loss))
    for i in range(len(out)):
        sl = slice(x.tok_off[i] + 1, x.tok_off[i] + x.n_tok[i])
        err = ~x.tok_top1[sl]
        d = x.tok_lp[sl][err].astype(np.float64) - base.tok_lp[sl][err].astype(np.float64)
        P, N = d[d > 0].sum(), -d[d < 0].sum()
        out[i] = P / N if err.sum() >= 2 and N > 0 else 0.0
    return out


# Offline RMIA, Low-Cost High-Power Membership Inference Attacks, https://arxiv.org/abs/2312.03262, with the
# pretrained model as the reference. Adapted: a is tuned on another seed's model instead of a paired reference model.
def rmia(pt, p_ref, iz, a):
    lr = pt / (0.5 * ((1 + a) * p_ref + 1 - a))
    lz = np.sort(lr[iz])
    return np.searchsorted(lz, lr / GAMMA, side="right") / len(lz)


def tune_a(p, p_base):
    """RMIA's a by AUC on the paired model p: its U (members) against its H, with its Z as population."""
    idx = np.r_[p.idx("U"), p.idx("H")]
    lab, pt, pr, iz = p.role[idx] == "U", p.prob(), p_base.prob(), p.idx("Z")
    return float(max(A_GRID, key=lambda a: auc(rmia(pt, pr, iz, a)[idx], lab)))


# Offline LiRA (one-sided, global variance), Membership Inference Attacks From First Principles,
# https://arxiv.org/abs/2112.03570. The global variance is a constant and does not change the ranking.
def lira_off(x, sibs, mask):
    ph = np.stack([s.phi() for s in sibs])
    n = mask.sum(0)
    return x.phi() - np.where(n > 0, (ph * mask).sum(0) / np.maximum(n, 1), np.nan)


def single_model(x, base, a, iz):
    # LOSS, Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting, https://arxiv.org/abs/1709.01604.
    # Calibrated loss, On the Importance of Difficulty Calibration in Membership Inference Attacks,
    # https://arxiv.org/abs/2111.08440, with the pretrained model as the reference.
    return {"A1_loss": -x.loss, "A2_calibrated": base.loss - x.loss, "A3_rmia_base": rmia(x.prob(), base.prob(), iz, a),
            "A4_mink_full": x.mink(False), "A4_minkpp_full": x.mink(True), "A5_zlib": x.zlib(),
            "A7_ez_full": ez(x, base)}


def check_aligned(a, b):
    same = all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("role", "k", "sha256", "n_tok", "opt_len"))
    if not (same and all(a.h[k] == b.h[k] for k in ("manifest_sha256", "model_name", "model_revision", "batch_tokens"))):
        raise RuntimeError(f"score files are not on the same records, tokens, base model or scoring path: "
                           f"{a.path} vs {b.path}")


def groups_of(x, field):
    man = C.load_manifest(x.h["task"], x.h["record_seed"])
    if man["_sha256"] != x.h["manifest_sha256"]:
        raise RuntimeError(f"{x.path} was scored on another manifest version")
    return np.array([man[r][int(k)][field] for r, k in zip(x.role, x.k)])


def out_mask(x, sibs):
    """(n_sibs, n_records): True where the sibling's U and dev hold neither the record nor its content group."""
    groups = groups_of(x, "group")
    rows = []
    for s in sibs:
        m = C.load_manifest(s.h["task"], s.h["train_set_seed"])
        seen = m["U"] + m["dev"]
        rows.append(~np.isin(x.row, [r["row"] for r in seen]) & ~np.isin(groups, [r["group"] for r in seen]))
    return np.array(rows, bool).reshape(len(sibs), len(x.row))


def load(d):
    ft, base = {}, {}
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(p) as z:
            h = json.loads(str(z["header"]))
        if h["state"] == "base":
            base[(h["task"], h["model_name"], h["record_seed"])] = Scores(p)
        else:
            ft[(h["run"]["run_id"], h["record_seed"])] = Scores(p)
    return ft, base


def analyse(key):
    run, seed = key
    x = FT[key]
    sib_runs = sorted(r for r, s in FT if s == seed and r != run and FT[(r, s)].h["run"]["family"] == x.h["run"]["family"])
    sibs = [FT[(r, seed)] for r in sib_runs]
    base = BASE[x.key()]
    for s in [base] + sibs:
        check_aligned(x, s)

    pair = min(sibs, key=lambda s: s.h["train_set_seed"])
    ps = pair.h["train_set_seed"]
    p, p_base = FT[(pair.h["run"]["run_id"], ps)], BASE[(x.h["task"], x.h["model_name"], ps)]
    check_aligned(p, p_base)
    a = tune_a(p, p_base)

    iu, iz = x.idx("U"), x.idx("Z")
    (j_star,) = x.h["run"]["logged_secret_ids"]
    lab = MAT[x.k[iu], j_star]
    mask = out_mask(x, sibs)
    att = single_model(x, base, a, iz)
    plac = [single_model(s, base, a, iz) for s in sibs]
    lab_b = base.label()
    att["A6_label"] = x.label()
    att["A8_lira_off"] = lira_off(x, sibs, mask)

    rng = np.random.default_rng([0, zlib.crc32(run.encode())])
    groups = groups_of(x, "article" if x.h["task"] == "SQuAD" else "group")[iu]
    common = mask[:, iu].all(0)   # records no sibling saw
    out = {}
    for name, score in att.items():
        s = score[iu]
        rs = resamples(groups, B, rng)
        ad = np.array([auc(s[i], lab[i]) for i in rs])
        r = {"auc": auc(s, lab), "auc_ci95": ci(ad), "tpr1": tpr_at_fpr(s, lab)}
        if name in plac[0]:
            sc, refs, lc = s[common], [pl[name][iu][common] for pl in plac], lab[common]
            stat = lambda i: auc(sc[i], lc[i]) - np.mean([auc(v[i], lc[i]) for v in refs])
            r.update(vs="other seeds", n_records=int(common.sum()), delta=stat(np.arange(len(lc))),
                     delta_ci95=ci([stat(i) for i in resamples(groups[common], B, rng)]))
        elif name == "A6_label":
            b = lab_b[iu]
            r.update(vs="pretrained", delta=r["auc"] - auc(b, lab),
                     delta_ci95=ci(ad - np.array([auc(b[i], lab[i]) for i in rs])))
        else:
            r.update(vs="0.5", delta=r["auc"] - 0.5, delta_ci95=[v - 0.5 for v in r["auc_ci95"]])
        r["lights_up"] = r["delta_ci95"][0] > 0
        out[name] = r
    return {"run": run, "family": x.h["run"]["family"], "task": x.h["task"],
            "model": x.h["model_name"], "lora": x.h["lora"], "seed": seed, "siblings": sib_runs,
            "rmia_paired": pair.h["run"]["run_id"], "rmia_a": a, "attacks": out}


def main():
    global FT, BASE, MAT, B
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    MAT, msha = C.load_matrix()
    FT, BASE = load(a.scores)
    B = a.boot
    keys = sorted((r, s) for (r, s), x in FT.items() if s == x.h["train_set_seed"] == SEED)
    with multiprocessing.get_context("fork").Pool(len(os.sched_getaffinity(0))) as pool:
        models = pool.map(analyse, keys, chunksize=1)
    with open(a.out, "w") as f:
        json.dump({"matrix_sha256": msha, "boot": B, "code": C.git_state(), "models": models}, f, indent=1)
    print(json.dumps({"out": a.out, "n_models": len(models)}))


if __name__ == "__main__":
    main()
