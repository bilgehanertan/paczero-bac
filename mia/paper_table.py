#!/usr/bin/env python3
import argparse, json

MODELS = [("1p3b", "OPT-1.3B"), ("6p7b", "OPT-6.7B"), ("qwen7b", "Qwen2.5-7B"), ("llama8b", "Llama-3.1-8B")]
TASKS = [("sst2", "SST-2"), ("squad", "SQuAD")]
ATTACKS = [("A1_loss", "LOSS"), ("A2_calibrated", "Loss calibrated by the pretrained model"), ("A3_rmia_base", "RMIA"),
           ("A4_mink_full", "Min-K\\%"), ("A4_minkpp_full", "Min-K\\%++"), ("A5_zlib", "zlib"),
           ("A6_label", "Label-only"), ("A7_ez_full", "EZ-MIA"), ("A8_lira_off", "Offline LiRA")]


def family(variant, task, model, track):
    return f"{task}_{model}_{track}_{variant}"


def sgn(x):
    return ("+" if round(x, 3) > 0 else ("-" if round(x, 3) < 0 else "")) + f"{abs(x):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    a = ap.parse_args()
    models = json.load(open(a.report))["models"]
    by_fam = {m["family"]: m for m in models}

    for task, tname in TASKS:
        for track in ("ft", "lora"):
            for model, mname in MODELS:
                cells = []
                for v in ("agc0", "bac0"):
                    m = by_fam.get(family(v, task, model, track))
                    x = m and m["attacks"]["A3_rmia_base"]
                    cells.append(f"${sgn(x['delta'])}$ $[{sgn(x['delta_ci95'][0])}, {sgn(x['delta_ci95'][1])}]$"
                                 if x else "--")
                if cells != ["--", "--"]:
                    print(f"{tname} & {'FT' if track == 'ft' else 'LoRA'} & {mname} & {cells[0]} & {cells[1]} \\\\")

    lit = sum(v["lights_up"] for m in models for v in m["attacks"].values())
    n = sum(len(m["attacks"]) for m in models)
    print(f"% checkpoints {len(models)}, tests {n}, lower bound above reference {lit} ({100 * lit / n:.1f}%)")
    auc = [m["attacks"]["A3_rmia_base"]["auc"] for m in models]
    print(f"% RMIA AUC {min(auc):.3f} to {max(auc):.3f}, mean {sum(auc) / len(auc):.3f}")

    print("% attack & AUC median [min, max] & TPR at FPR <= 1% mean [max] over all checkpoints")
    for key, label in ATTACKS:
        v = [m["attacks"][key] for m in models]
        auc = sorted(x["auc"] for x in v)
        med = (auc[(len(auc) - 1) // 2] + auc[len(auc) // 2]) / 2
        tpr = [100 * x["tpr1"]["tpr"] for x in v]
        # "--": the score takes so few values that FPR <= 1% is reached only at FPR 0, on every checkpoint
        cell = "--" if all(x["tpr1"]["achieved_fpr"] == 0 for x in v) else f"${sum(tpr) / len(tpr):.1f}$ $[{max(tpr):.1f}]$"
        print(f"{label} & ${med:.3f}$ $[{auc[0]:.3f}, {auc[-1]:.3f}]$ & {cell} \\\\")


if __name__ == "__main__":
    main()
