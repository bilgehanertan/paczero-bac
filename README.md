# PACZero

Code for *PACZero: PAC-Private Fine-Tuning of Language Models via Sign Quantization* (NeurIPS 2026).

PACZero fine-tunes language models with sign-quantized zeroth-order updates. It supports full fine-tuning and
LoRA on SST-2 and SQuAD, with Gaussian and binary channels for controlling information leakage.

## Methods

| Paper name | Variant in `configs/variants.yaml` | Mechanism keys |
|---|---|---|
| PACZero-AGC | `agc` | `pac_channel: gaussian`, `pac_mi` per run |
| PACZero-BAC | `bac` | `pac_channel: binary`, `pac_mi` per run |
| PACZero-AGC_0 | `agc0` | `pac_zero_budget: coin` |
| PACZero-BAC_0 | `bac0` | `pac_zero_budget: majority` |

`pac_mi` sets the total mutual-information budget. AGC adds Gaussian noise before re-quantization. BAC uses
a binary asymmetric channel. At zero budget, AGC_0 releases a fair coin when the subsets disagree, while BAC_0
releases the posterior-majority sign.

`pac_k` sets the number of perturbation directions per step. The `nonprivate` variant is the non-private ZO
reference. The `dpzero` and `dpaggzo` variants run the DPZero and DP-AggZO baselines.

## Environment

Use Python 3.12 and a CUDA GPU. Package versions are pinned in `requirements.txt`.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
bash setup.sh
```

`setup.sh` clones DP-AggZO at a fixed commit, resets it to that commit and applies the patches in `baselines/`.

## Models and data

The models are OPT-1.3B, OPT-6.7B, Qwen2.5-7B and Llama-3.1-8B. The tasks are SST-2 and SQuAD v1.1.
For Llama runs, accept the model license on Hugging Face and sign in with `huggingface-cli login`.

Download a model and both datasets at the paper's revisions, listed in `assets.json`:

```bash
python prepare_assets.py facebook/opt-1.3b
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

## Reproducing a run

Every reported run is one line of a YAML file under `configs/`. `CONFIGS.md` maps each paper table and figure
to its files. Run from the repository root:

```bash
python -m paczo.run configs/main/grid.yaml --run sst2_1p3b_lora_agc_mi033_s0
```

`--list` prints the run ids of a file. `--dry-run` prints the resolved configuration and exits. Every run prints
it before it starts, with the source of each value the configs set and then every argument, defaults included.

A run line names its setting and variant. The setting supplies the model, task, track, attention kernel, learning
rate, clipping threshold, step budget and, for SQuAD, the trainer switches of `configs/settings.yaml`. The variant
supplies the mechanism. Keys on the run line override the setting or the shared protocol, which is how a sweep or an
ablation differs from the main grid. DP and zero-shot runs take no setting.

PACZero runs write metrics to `result/runs/<run_id>/`, and training runs also write per-step diagnostics there and
checkpoints to `checkpoints/<run_id>/` (set `PAC_CKPT_BASE` to change it). Each training run keeps the checkpoint with
the lowest development loss and reports its test score. The timing runs of `configs/cost/timing.yaml` end before
their first development evaluation and report the final model, and the zero-shot runs evaluate the pretrained model.
The DPZero and DP-AggZO baselines write their checkpoints to `result/runs/<run_id>/` and their metrics to
`baselines/dp-aggzo/opt/result/`.

Set `PAC_NOISE_SEED` to repeat the private random draws.

## Membership inference

`mia/` contains the membership-inference evaluation of the paper's appendix: nine black-box attacks (LOSS, loss
calibrated by the pretrained model, offline RMIA with the pretrained reference, Min-K%, Min-K%++, zlib, label-only,
EZ-MIA, offline LiRA) on the dev-selected seed-2 checkpoints of the PACZero-AGC_0 and PACZero-BAC_0 runs in
`configs/main/grid.yaml`. The runs use the protocol's sizes (1000 training and 500 dev records, 128 candidate
subsets) and the default `seed` of 42, which fixes the candidate matrix. `mia/manifests/` holds the frozen records of
every task and seed and that matrix. Each setting needs its runs at the other seeds of the grid: the attacks are
compared with the other seeds' models, RMIA tunes its offline coefficient on one of them, and offline LiRA uses them
as references. The pretrained reference is built from a LoRA run of the same model, task and seed, whose checkpoint
carries the frozen base weights.

Train a setting at every seed of the grid:

```bash
for s in 0 1 2; do python -m paczo.run configs/main/grid.yaml --run sst2_1p3b_lora_agc0_s$s; done
```

Check that a run drew the frozen candidate matrix. The script replays the run to step 0 with the trainer's
membership dump (`pac_dump_membership`) and fails on any difference:

```bash
python mia/dump_membership.py --ckpt-dir checkpoints/sst2_1p3b_lora_agc0_s2/checkpoint-<step> --out dump_s2.npz
```

Score each run on its own records and on the other seeds' records, and score the pretrained model on each
seed's records. Then run the attacks:

```bash
for s in 0 1 2; do
  run=sst2_1p3b_lora_agc0_s$s
  python mia/score_records.py --run result/runs/$run --also-seeds 0,1,2 --out mia_scores/$run.ft.npz
  python mia/score_records.py --run result/runs/$run --base --out mia_scores/base_opt-1.3b_SST2_s$s.npz
done
python mia/analyse.py --scores mia_scores --out mia_scores/report.json
```

`analyse.py` attacks the seed-2 checkpoints. With every setting of the grid scored,
`python mia/paper_table.py mia_scores/report.json` prints the appendix tables.

## Code

`paczo/` contains the trainer, the config loader, the entry point and the information-budget calculations.
`src/` contains data loading, prompts, losses, LoRA and metrics. `configs/` contains the experiment recipes.
`mia/` contains the membership-inference evaluation.

The supporting code builds on [DP-AggZO](https://github.com/erguteb/dp-aggzo), DPZero and
[MeZO](https://github.com/princeton-nlp/MeZO). LoRA follows [loralib](https://github.com/microsoft/LoRA).

## License

MIT. See `LICENSE` and the upstream notices in `LICENSES/`.
