import glob, hashlib, json, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MANIFEST_DIR = os.path.join(REPO, "mia", "manifests")

with open(os.path.join(REPO, "assets.json")) as _f:
    _ASSETS = json.load(_f)
PINNED = _ASSETS["models"]
DATASET_REVISIONS = {k: (v["path"], v["name"], v["revision"]) for k, v in _ASSETS["datasets"].items()}


def pin_datasets():
    """Offline, `datasets` 2.21 ignores `revision` and serves whatever the cache holds, so the cache directory the
    arrow files come from is asserted to be the pinned revision."""
    import datasets
    import src.tasks
    pinned = {(p, n): r for p, n, r in DATASET_REVISIONS.values()}

    def load(path, name=None, **kw):
        rev = pinned[(path, name)]
        d = datasets.load_dataset(path, name, revision=rev, **kw)
        for split, ds in d.items():
            files = [f["filename"] for f in ds.cache_files]
            if not files or not all(f"{os.sep}{rev}{os.sep}" in f for f in files):
                raise RuntimeError(f"{path}/{name} {split} not read from the pinned revision {rev}: {files[:1]}")
        return d
    src.tasks.load_dataset = load


def full_fingerprint(task):
    out = {}
    for split in ("train", "valid"):
        h = hashlib.sha256()
        for s in task.samples[split]:
            h.update(json.dumps([s.id, s.data, s.candidates, s.correct_candidate], sort_keys=True,
                                ensure_ascii=False).encode("utf-8"))
            h.update(b"\n")
        out[split] = h.hexdigest()
    return out


def run_family(run_id):
    return re.sub(r"_s[0-9]+$", "", run_id)


def best_checkpoint(ckpt_base):
    last = max(glob.glob(os.path.join(ckpt_base, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    best = json.load(open(os.path.join(last, "trainer_state.json")))["best_model_checkpoint"]
    return os.path.join(ckpt_base, os.path.basename(best))


def logged_secret_ids(diag_path):
    ids = set()
    with open(diag_path) as f:
        for line in f:
            m = re.search(r'"pac_secret_id":\s*(-?\d+)', line)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_state():
    if not os.path.isdir(os.path.join(REPO, ".git")):
        return {"commit": None, "source": "exported tree"}
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", "mia", "paczo", "src"],
                                capture_output=True, text=True).stdout.strip())
    return {"commit": head, "dirty_code": dirty}


def gold_text(task, sample):
    return task.get_template().verbalize(sample, sample.correct_candidate).strip(" ")


def load_manifest(task_name, seed):
    path = os.path.join(MANIFEST_DIR, f"{task_name}_s{seed}.json")
    with open(path) as f:
        m = json.load(f)
    m["_path"], m["_sha256"] = path, sha256_file(path)
    return m


def load_matrix():
    import numpy as np
    path = os.path.join(MANIFEST_DIR, "matrix_seed42_M128.npz")
    return np.load(path)["membership"], sha256_file(path)


def resolve_records(task, manifest, roles):
    split = {"U": "train", "dev": "train", "H": "train", "Z": "train", "test": "valid"}
    out = []
    for role in roles:
        samples = task.samples[split[role]]
        for k, r in enumerate(manifest[role]):
            s = samples[r["row"]]
            if int(s.id) != r["id"] or sha256_text(gold_text(task, s)) != r["sha256"]:
                raise RuntimeError(f"record mismatch: {manifest['task']} s{manifest['train_set_seed']} "
                                   f"{role}[{k}] row {r['row']}: dataset differs from the manifest")
            out.append((role, k, r, s))
    return out


def load_training_args(path):
    import torch
    return torch.load(path, weights_only=False)


def resolve_model_path(model_name):
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name, revision=PINNED[model_name], ignore_patterns=["original/*"],
                             allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model", "tokenizer*"])


def assert_on_gpu(model):
    # device_map="auto" offloads blocks that do not fit, and load_state_dict into an offloaded block is a no-op.
    bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type != "cuda"]
    off = {k: v for k, v in (getattr(model, "hf_device_map", None) or {}).items()
           if not (isinstance(v, int) or str(v).startswith("cuda"))}
    if bad or off:
        raise RuntimeError(f"model not fully on the GPU: {bad[:3]} {dict(list(off.items())[:3])}")


def assert_base_matches(model, ckdir):
    import torch
    sd, _ = read_state_dict(ckdir)
    n, missing = 0, []
    for k, v in model.state_dict().items():
        if k not in sd:
            missing.append(k); continue
        if not torch.equal(v, sd[k].to(device=v.device, dtype=v.dtype)) or v.dtype != sd[k].dtype:
            raise RuntimeError(f"base tensor {k} differs from the frozen weights in {ckdir}")
        n += 1
    tied = getattr(model.config, "tie_word_embeddings", False)
    if [k for k in missing if not (tied and k.endswith("lm_head.weight"))]:
        raise RuntimeError(f"base tensors absent from the LoRA checkpoint: {missing[:3]}")
    return n


def read_state_dict(ckdir):
    import torch
    idx_st, idx_bin = os.path.join(ckdir, "model.safetensors.index.json"), os.path.join(ckdir, "pytorch_model.bin.index.json")
    if os.path.isfile(idx_st) or os.path.isfile(idx_bin):
        idx = idx_st if os.path.isfile(idx_st) else idx_bin
        files = sorted(set(json.load(open(idx))["weight_map"].values()))
    elif os.path.isfile(os.path.join(ckdir, "model.safetensors")):
        files = ["model.safetensors"]
    else:
        files = ["pytorch_model.bin"]
    sd = {}
    for fn in files:
        p = os.path.join(ckdir, fn)
        if fn.endswith(".safetensors"):
            from safetensors.torch import load as st_load
            with open(p, "rb") as f:
                part = st_load(f.read())
        else:
            part = torch.load(p, map_location="cpu", weights_only=True)
        sd.update(part)
    return sd, files


def load_checkpoint_strict(model, ckdir):
    import time
    import torch
    t0 = time.time()
    sd, files = read_state_dict(ckdir)
    t1 = time.time()
    res = model.load_state_dict(sd, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    tied = set()
    if getattr(model.config, "tie_word_embeddings", False):
        tied = {k for k in missing if k.endswith("lm_head.weight")}
    if unexpected:
        raise RuntimeError(f"checkpoint has keys the model does not: {unexpected[:5]}")
    if set(missing) - tied:
        raise RuntimeError(f"model keys absent from the checkpoint: {sorted(set(missing) - tied)[:5]}")
    if tied and model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise RuntimeError("lm_head is not tied to the input embeddings after loading")
    lora_model = sorted(k for k, _ in model.named_parameters() if "lora_" in k)
    lora_ckpt = sorted(k for k in sd if "lora_" in k)
    if lora_model != lora_ckpt:
        raise RuntimeError(f"LoRA tensors differ: model {len(lora_model)}, checkpoint {len(lora_ckpt)}")
    msd = model.state_dict()
    for k, v in sd.items():
        t = msd[k]
        if t.device.type == "meta" or not torch.equal(t, v.to(device=t.device, dtype=t.dtype)):
            raise RuntimeError(f"{k} does not hold the checkpoint's values after loading ({t.device})")
    return {"files": files, "n_tensors": len(sd), "missing_tied": sorted(tied), "n_lora": len(lora_ckpt),
            "read_s": round(t1 - t0, 1), "load_and_verify_s": round(time.time() - t1, 1)}
