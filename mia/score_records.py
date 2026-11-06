#!/usr/bin/env python3
import argparse, json, os, platform, sys, tempfile, time, zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ROLES = ("U", "H", "Z")
S1_TOL = {"SST2": 0.004, "SQuAD": 0.005}   # 2 of 500 dev examples; 0.5 F1 points
S1B_TOL = 2e-3                               # nats; fp16 spacing near 0.26 is 2.4e-4, batch padding adds about that
S1D_MIN = 0.95                               # batched vs batch-1 agreement on dev records
CHUNK = 256                                  # positions per vocabulary reduction


def _chunk_stats(lg, y, out, torch):
    lg = lg.float()
    top1 = lg.argmax(-1) == y
    lp = torch.log_softmax(lg, dim=-1)
    p = lp.exp()
    mu = (p * lp).sum(-1)
    # Two-pass variance: E[x^2] - E[x]^2 cancels catastrophically in float32 when the distribution is flat.
    var = (p * (lp - mu[:, None]) ** 2).sum(-1)
    out["lp"].append(lp.gather(-1, y[:, None])[:, 0])
    out["mu"].append(mu)
    out["sd"].append(var.clamp_min(0).sqrt())
    out["top1"].append(top1)


def length_batches(lens, budget, extra=0):
    order = sorted(range(len(lens)), key=lambda i: (-lens[i], i))
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * (lens[cur[0]] + extra) > budget:
            batches.append(cur); cur = []
        cur.append(i)
    return batches + ([cur] if cur else [])


def batch_token_stats(model, seqs, pad_id, budget, torch):
    """Per-position statistics of every next-token prediction. Right padding leaves every real position's logits
    those of the unpadded sequence in a causal model (position ids 0..n-1 as in batch 1)."""
    res = [None] * len(seqs)
    for batch in length_batches([len(q) for q in seqs], budget):
        n = torch.tensor([len(seqs[i]) for i in batch])
        x = torch.full((len(batch), int(n.max())), pad_id, dtype=torch.long)
        for b, i in enumerate(batch):
            x[b, :n[b]] = torch.tensor(seqs[i])
        mask = (torch.arange(x.shape[1])[None, :] < n[:, None]).long()
        x, mask = x.to(model.device), mask.to(model.device)
        with torch.inference_mode():
            logits = model(input_ids=x, attention_mask=mask).logits
        bi, ti = (torch.arange(x.shape[1] - 1)[None, :] < (n - 1)[:, None]).nonzero(as_tuple=True)
        bi, ti = bi.to(model.device), ti.to(model.device)
        lab = x[bi, ti + 1]
        out = {k: [] for k in ("lp", "mu", "sd", "top1")}
        for c in range(0, bi.numel(), CHUNK):
            _chunk_stats(logits[bi[c:c + CHUNK], ti[c:c + CHUNK]], lab[c:c + CHUNK], out, torch)
        cuts = np.cumsum((n - 1).numpy())[:-1]
        parts = {k: np.split(torch.cat(v).cpu().numpy(), cuts) for k, v in out.items()}
        if not all(np.isfinite(np.concatenate(parts[k])).all() for k in ("lp", "mu", "sd")):
            # a non-finite value at a pad position would reach the real positions of its row through attention
            raise RuntimeError("non-finite token statistics in a padded batch")
        for b, i in enumerate(batch):
            res[i] = {k: parts[k][b] for k in parts}
        del logits
    return res


def batch_greedy_decode(model, tok, prompts, targs, budget, torch):
    """Framework.forward(generation=True) for many prompts: the same generate() arguments and stop tokens, left-padded,
    batched only with prompts of the same length cap so no row is generated past its own cap."""
    eos = [tok.encode(targs.eos_token, add_special_tokens=False)[-1], tok.eos_token_id]
    texts = [None] * len(prompts)
    cap_of = [min(targs.max_new_tokens, targs.max_length - len(q)) for q in prompts]
    batches = []
    for cap in sorted(set(cap_of), reverse=True):
        idx = [i for i in range(len(prompts)) if cap_of[i] == cap]
        batches += [[idx[j] for j in b] for b in length_batches([len(prompts[i]) for i in idx], budget, extra=cap)]
    for batch in batches:
        L = max(len(prompts[i]) for i in batch)
        caps = [cap_of[i] for i in batch]
        x = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for b, i in enumerate(batch):
            x[b, L - len(prompts[i]):] = torch.tensor(prompts[i]); mask[b, L - len(prompts[i]):] = 1
        with torch.inference_mode():
            out = model.generate(x.to(model.device), attention_mask=mask.to(model.device), do_sample=targs.sampling,
                                 temperature=targs.temperature, num_beams=targs.num_beams, top_p=targs.top_p,
                                 top_k=targs.top_k, max_new_tokens=max(caps), num_return_sequences=1,
                                 eos_token_id=eos, pad_token_id=tok.pad_token_id)
        for b, i in enumerate(batch):
            g = out[b, L:L + caps[b]].tolist()
            stop = next((j for j, t in enumerate(g) if t in eos), None)
            texts[i] = tok.decode(g if stop is None else g[:stop + 1], skip_special_tokens=True).strip()
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the run's result directory, result/runs/<run_id>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", action="store_true", help="score the pretrained model, built with this LoRA run's recipe")
    ap.add_argument("--batch-tokens", type=int, default=8192, help="padded tokens per scoring and decoding batch")
    ap.add_argument("--also-seeds", default="", help="also score the records of these train_set_seeds, written to "
                    "<out>.rs<seed>.npz")
    a = ap.parse_args()

    import torch
    import transformers
    from src.tasks import get_task
    from src.utils import Prediction, encode_prompt
    from src.metrics import calculate_metric
    from paczo.run_paczo import Framework

    t0 = time.time()
    run_dir = os.path.normpath(a.run)
    run_id = os.path.basename(run_dir)
    ckpt_dir = C.best_checkpoint(os.path.join(os.environ.get("PAC_CKPT_BASE", os.path.join(C.REPO, "checkpoints")), run_id))
    ckpt = os.path.basename(ckpt_dir)
    ident = {"run_id": run_id, "family": C.run_family(run_id), "checkpoint": ckpt,
             "result_dir": os.path.abspath(run_dir),
             "logged_secret_ids": C.logged_secret_ids(os.path.join(run_dir, "checkpoints", "diagnostics.jsonl"))}
    targs = C.load_training_args(os.path.join(ckpt_dir, "training_args.bin"))
    task_name, seed = targs.task_name, int(targs.train_set_seed)
    tmp = tempfile.mkdtemp(prefix="mia_score_")
    targs.output_dir = targs.logging_dir = targs.pac_diag_dir = tmp
    targs.result_file, targs.run_name = os.path.join(tmp, "metrics.json"), "mia_score"
    model_name = targs.model_name
    targs.model_name = C.resolve_model_path(model_name)
    if a.base:
        targs.lora = False

    C.pin_datasets()
    task = get_task(task_name)
    man = C.load_manifest(task_name, seed)
    if C.full_fingerprint(task) != man["dataset_fingerprint_full"]:
        sys.exit("dataset fingerprint differs from the manifest: wrong dataset revision")

    fw = Framework(targs, task)
    model, tok = fw.model, fw.tokenizer
    C.assert_on_gpu(model)
    load_info = s2 = None
    if a.base:
        s2 = {"recipe_run": run_id, "n_tensors_equal": C.assert_base_matches(model, ckpt_dir)}
    else:
        load_info = C.load_checkpoint_strict(model, ckpt_dir)
    model.eval()
    tok.padding_side = "left"   # as Framework.train leaves it; the batched paths pad explicitly

    header = {
        "schema": "paczero-mia-scores/2", "state": "base" if a.base else "finetuned",
        "model_name": model_name, "model_revision": C.PINNED[model_name], "model_path": targs.model_name,
        "task": task_name, "train_set_seed": seed, "lora": bool(targs.lora),
        "lora_r": targs.lora_r, "lora_alpha": targs.lora_alpha, "load_float16": bool(targs.load_float16),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "max_length": targs.max_length, "max_new_tokens": targs.max_new_tokens,
        "run": ident, "ckpt_dir": os.path.abspath(ckpt_dir), "load": load_info, "s2": s2, "s1": None,
        "batch_tokens": a.batch_tokens,
        "code": C.git_state(), "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "host": platform.node(), "python": sys.version.split()[0],
    }
    score = lambda m, roles, decode: score_set(task, m, model, tok, targs, torch, encode_prompt, calculate_metric,
                                               Prediction, roles, decode, a.batch_tokens)

    if not a.base:
        metric = getattr(task, "metric_name", "accuracy")
        dev = [s for _, _, _, s in C.resolve_records(task, man, ["dev"])]
        test = [s for _, _, _, s in C.resolve_records(task, man, ["test"])]
        dev_pred = [fw.one_step_pred(s) for s in dev]
        rec = json.load(open(os.path.join(run_dir, "metrics.json")))
        s1 = {"metric": "dev_" + metric, "recomputed": float(calculate_metric(dev_pred, metric)),
              "recorded": rec["dev_" + metric], "test_recomputed": float(fw.evaluate(test)[metric]),
              "test_recorded": rec[metric], "tol": S1_TOL[task_name]}
        s1["abs_diff"] = abs(s1["recomputed"] - s1["recorded"])
        s1["test_abs_diff"] = abs(s1["test_recomputed"] - s1["test_recorded"])
        step = int(ckpt.rsplit("-", 1)[1])
        st = json.load(open(os.path.join(ckpt_dir, "trainer_state.json")))
        ev = [h["eval_loss"] for h in st["log_history"] if h.get("step") == step and "eval_loss" in h][-1]
        dev_arr = score(man, ["dev"], False)
        s1.update(dev_loss_recomputed=float(np.mean(dev_arr["loss"])), eval_loss_recorded=ev)
        s1["dev_loss_abs_diff"] = abs(s1["dev_loss_recomputed"] - ev)
        if task.generation:
            prompts = [encode_prompt(task.get_template(), s, tok, max_length=targs.max_length,
                                     generation=True, max_new_tokens=targs.max_new_tokens)[0][0] for s in dev]
            got = batch_greedy_decode(model, tok, prompts, targs, a.batch_tokens, torch)
        else:
            got = [int(x) for x in dev_arr["pred"]]
        agree = sum(g == p.predicted_candidate for g, p in zip(got, dev_pred))
        s1["batched_agreement"] = {"n": len(dev), "agree": int(agree), "min_rate": S1D_MIN}
        # 1e-9: a difference of exactly tol (2 of 500 records) passes despite rounding in the subtraction
        s1["pass"] = bool(s1["abs_diff"] <= s1["tol"] + 1e-9 and s1["test_abs_diff"] <= s1["tol"] + 1e-9
                          and s1["dev_loss_abs_diff"] <= S1B_TOL and agree / len(dev) >= S1D_MIN)
        header["s1"] = s1
        print("S1", json.dumps(s1), flush=True)
        if not s1["pass"]:
            sys.exit("S1 failed: this checkpoint does not reproduce its recorded metrics")
    t_s1 = time.time()

    for rs in [seed] + [int(x) for x in a.also_seeds.split(",") if x != "" and int(x) != seed]:
        man_rs = man if rs == seed else C.load_manifest(task_name, rs)
        out = a.out if rs == seed else a.out[:-4] + f".rs{rs}.npz"
        t1 = time.time()
        arrays = score(man_rs, ROLES, rs == seed and task.generation)
        h = dict(header, record_seed=rs, own_seed=(rs == seed), manifest=os.path.basename(man_rs["_path"]),
                 manifest_sha256=man_rs["_sha256"], gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                 timing_s={"setup_and_s1": round(t_s1 - t0, 1), "scoring": round(time.time() - t1, 1)})
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez_compressed(out + ".tmp.npz", header=np.array(json.dumps(h)), **arrays)
        os.replace(out + ".tmp.npz", out)
        print(json.dumps({"out": out, "record_seed": rs, "n_records": len(arrays["k"]), "timing_s": h["timing_s"]}),
              flush=True)


def score_set(task, man, model, tok, targs, torch, encode_prompt, calculate_metric, Prediction, roles, decode, bt):
    """Gold-verbalized encodings as the trainer trains on them. SST-2: both candidates, loss = 2-way CE of their mean
    option log-probs. SQuAD: loss = mean answer-span NLL, and with `decode` the greedy-decode F1 of U."""
    recs = C.resolve_records(task, man, roles)
    gen = task.generation
    enc = [encode_prompt(task.get_template(), s, tok, max_length=targs.max_length, generation=True,
                         generation_with_gold=True, max_new_tokens=targs.max_new_tokens) if gen else
           encode_prompt(task.get_template(), s, tok, max_length=targs.max_length, generation=False)
           for _, _, _, s in recs]
    flat_st = batch_token_stats(model, [e for encs, _ in enc for e in encs], tok.pad_token_id, bt, torch)
    stats, j = [], 0
    for encs, _ in enc:
        stats.append(flat_st[j:j + len(encs)]); j += len(encs)
    di = [i for i, rec in enumerate(recs) if decode and rec[0] == "U"]
    prompts = [encode_prompt(task.get_template(), recs[i][3], tok, max_length=targs.max_length,
                             generation=True, max_new_tokens=targs.max_new_tokens)[0][0] for i in di]
    texts = dict(zip(di, batch_greedy_decode(model, tok, prompts, targs, bt, torch)))
    cols = {k: [] for k in ("role", "k", "row", "id", "sha256", "y", "opt_len", "n_tok", "loss", "pred",
                            "dec_f1", "zlib_len")}
    cand_lp, flat = [], {k: [] for k in ("lp", "mu", "sd", "top1")}
    for idx, ((role, k, r, s), (encs, olens)) in enumerate(zip(recs, enc)):
        if gen:
            st, ol = stats[idx][0], olens[0]
            y, loss, pred, cl = -1, float(-st["lp"][-ol:].mean()), -1, [np.nan, np.nan]
            f1 = float(calculate_metric([Prediction(correct_candidate=s.correct_candidate,
                                                    predicted_candidate=texts[idx])], "f1")) if idx in texts else np.nan
        else:
            y = s.candidates.index(s.correct_candidate)
            per = stats[idx]
            cl = [float(p["lp"][-o:].mean()) for p, o in zip(per, olens)]
            st, ol = per[y], olens[y]
            lse = np.logaddexp.reduce(np.array(cl, dtype=np.float64))
            loss, pred, f1 = float(lse - cl[y]), int(np.argmax(cl)), np.nan
        for kk in flat:
            flat[kk].append(st[kk])
        cand_lp.append(cl)
        for kk, v in (("role", role), ("k", k), ("row", r["row"]), ("id", r["id"]), ("sha256", r["sha256"]),
                      ("y", y), ("opt_len", ol), ("n_tok", len(st["lp"])), ("loss", loss), ("pred", pred),
                      ("dec_f1", f1), ("zlib_len", len(zlib.compress(C.gold_text(task, s).encode("utf-8"))))):
            cols[kk].append(v)
    n_tok = np.array(cols["n_tok"], dtype=np.int64)
    return dict(
        role=np.array(cols["role"]), k=np.array(cols["k"], dtype=np.int64), row=np.array(cols["row"], dtype=np.int64),
        id=np.array(cols["id"], dtype=np.int64), sha256=np.array(cols["sha256"]), y=np.array(cols["y"], dtype=np.int64),
        opt_len=np.array(cols["opt_len"], dtype=np.int64), n_tok=n_tok,
        tok_off=np.concatenate([[0], np.cumsum(n_tok)]).astype(np.int64),
        loss=np.array(cols["loss"], dtype=np.float64), pred=np.array(cols["pred"], dtype=np.int64),
        dec_f1=np.array(cols["dec_f1"], dtype=np.float64), zlib_len=np.array(cols["zlib_len"], dtype=np.int64),
        cand_lp=np.array(cand_lp, dtype=np.float64),
        tok_lp=np.concatenate(flat["lp"]).astype(np.float32), tok_mu=np.concatenate(flat["mu"]).astype(np.float32),
        tok_sd=np.concatenate(flat["sd"]).astype(np.float32), tok_top1=np.concatenate(flat["top1"]).astype(bool),
    )


if __name__ == "__main__":
    main()
