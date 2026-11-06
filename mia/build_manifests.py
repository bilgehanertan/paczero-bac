#!/usr/bin/env python3
"""Freeze the records and the candidate matrix the MIA uses: per task and train_set_seed, U and dev as
task.sample_train_set draws them, test as task.sample_subset draws it, and two never-touched pools from the rest of
the permutation, deduplicated against U and dev (rule in `pool_scan`): H (non-members of the paired model that tunes
RMIA's a) and Z (the RMIA population). The matrix is the trainer's: np.random.seed(42), then
np.random.choice(128, 64, replace=False) per U record.
"""
import argparse, hashlib, json, os, sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "mia"))
import common as C
C.pin_datasets()
from src.tasks import get_task
from src.utils import temp_seed

N_U, N_DEV, N_TEST, N_POOL = 1000, 500, 1000, 1000
M, MECH_SEED, J_STAR = 128, 42, 102
SEEDS = (0, 1, 2)
TASKS = ("SST2", "SQuAD")


def group_of(task_name, s):
    key = s.data["context"] if task_name == "SQuAD" else s.data["sentence"].strip().lower()
    return C.sha256_text(key)[:16]


def rec(task, samples, row, task_name):
    s = samples[row]
    r = {"row": int(row), "id": int(s.id), "sha256": C.sha256_text(C.gold_text(task, s)), "group": group_of(task_name, s)}
    if task_name == "SQuAD":
        r["article"] = C.sha256_text(s.data["title"])[:16]
    return r


def build_matrix():
    np.random.seed(MECH_SEED)
    mat = np.zeros((N_U, M), dtype=bool)
    for i in range(N_U):
        mat[i, np.random.choice(M, M // 2, replace=False)] = True
    assert (mat.sum(1) == M // 2).all()
    return mat


def dedup_sst2(u_samples):
    norm = lambda t: " " + " ".join(t.strip().lower().split()) + " "   # whole-word containment on spaces
    u = [norm(s.data["sentence"]) for s in u_samples]
    exact = set(u)
    long_u = [x for x in u if len(x.split()) >= 4]

    def drop(sample):
        h = norm(sample.data["sentence"])
        if h in exact:
            return "exact"
        for x in long_u:
            if x in h or h in x:
                return "containment"
        return None
    return drop


def dedup_squad(u_samples):
    ctx = {s.data["context"] for s in u_samples}

    def drop(sample):
        return "shared_context" if sample.data["context"] in ctx else None
    return drop


def build(task_name, task, seed):
    train, valid = task.samples["train"], task.samples["valid"]
    with temp_seed(seed):
        perm = np.random.permutation(len(train)).tolist()
    with temp_seed(seed):
        tperm = np.random.permutation(len(valid)).tolist()[:N_TEST]
    ref = task.sample_train_set(num_train=N_U, num_dev=N_DEV, seed=seed)
    assert [s.id for s in ref] == [train[i].id for i in perm[: N_U + N_DEV]], "sample_train_set mismatch"
    ref_test = task.sample_subset(data_split="valid", seed=seed, num=N_TEST)
    assert [s.id for s in ref_test] == [valid[i].id for i in tperm], "test draw mismatch"

    u_rows, dev_rows = perm[:N_U], perm[N_U : N_U + N_DEV]
    drop = (dedup_sst2 if task_name == "SST2" else dedup_squad)([train[i] for i in u_rows + dev_rows])
    pool, dropped = [], {}
    for pos in range(N_U + N_DEV, len(perm)):
        why = drop(train[perm[pos]])
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        pool.append((pos, perm[pos]))
        if len(pool) == 2 * N_POOL:
            last_pos = pos
            break
    else:
        raise RuntimeError("train pool exhausted before H and Z were filled")
    h, z = pool[:N_POOL], pool[N_POOL:]
    return {
        "task": task_name, "train_set_seed": seed, "n_train_split": len(train), "n_valid_split": len(valid),
        "mech_seed": MECH_SEED, "M": M, "j_star": J_STAR,
        "manifest_version": 3, "dataset": list(C.DATASET_REVISIONS[task_name]),
        "dataset_fingerprint_train": C.sha256_text("\n".join(C.gold_text(task, s) for s in train)),
        "dataset_fingerprint_full": C.full_fingerprint(task),
        "U": [rec(task, train, r, task_name) for r in u_rows],
        "dev": [rec(task, train, r, task_name) for r in dev_rows],
        "test": [rec(task, valid, r, task_name) for r in tperm],
        "H": [dict(rec(task, train, r, task_name), perm_pos=p) for p, r in h],
        "Z": [dict(rec(task, train, r, task_name), perm_pos=p) for p, r in z],
        "pool_scan": {"first_pos": N_U + N_DEV, "last_pos": last_pos, "dropped": dropped,
                      "rule": "SST2: exact, or whole-word containment either way with a U or dev sentence of >= 4 words "
                              "(lower-cased, whitespace-normalised); "
                              "SQuAD: context paragraph appears in U or dev"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "mia", "manifests"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sums = {}

    mat = build_matrix()
    mpath = os.path.join(a.out, f"matrix_seed{MECH_SEED}_M{M}.npz")
    np.savez_compressed(mpath, membership=mat, j_star=np.int64(J_STAR), mech_seed=np.int64(MECH_SEED))
    packed = hashlib.sha256(np.packbits(mat).tobytes()).hexdigest()
    print(f"matrix: column {J_STAR} has {int(mat[:, J_STAR].sum())} members; sha256(packbits) {packed}")

    for name in TASKS:
        task = get_task(name)
        for s in SEEDS:
            m = build(name, task, s)
            m["matrix_file"] = os.path.basename(mpath)
            m["matrix_packbits_sha256"] = packed
            m["n_members_g1"] = int(mat[:, J_STAR].sum())
            p = os.path.join(a.out, f"{name}_s{s}.json")
            with open(p, "w") as f:
                json.dump(m, f, indent=0)
            print(f"{name} s{s}: U {len(m['U'])} dev {len(m['dev'])} test {len(m['test'])} "
                  f"H {len(m['H'])} Z {len(m['Z'])} dropped {m['pool_scan']['dropped']} "
                  f"U[0:3] rows {[r['row'] for r in m['U'][:3]]}", flush=True)

    for fn in sorted(os.listdir(a.out)):
        if fn != "SHA256SUMS":
            sums[fn] = C.sha256_file(os.path.join(a.out, fn))
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as f:
        f.writelines(f"{v}  {k}\n" for k, v in sums.items())


if __name__ == "__main__":
    main()
