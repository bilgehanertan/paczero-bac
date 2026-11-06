# Configs behind each paper table and figure

Each line of an experiment file is one run, and its `id` is the run id of the results. `configs/settings.yaml` holds the model, task, track, attention kernel, learning rate, clipping threshold, step budget and SQuAD trainer switches of every setting (`tab:pool-recipes`). `configs/variants.yaml` holds the shared protocol and the mechanism of each method.

| Paper table or figure | Label | Config files |
|---|---|---|
| Main results | `tab:headline` | `main/grid.yaml`, `references/nonprivate.yaml`, `sweeps/reference_learning_rate.yaml`, `dp/baselines.yaml` |
| DP baseline hyperparameters and the Llama-3.1-8B learning-rate selection | `tab:headline-budget` | `dp/baselines.yaml`, `dp/learning_rate.yaml` |
| Hyperparameters of every setting | `tab:pool-recipes` | `settings.yaml` |
| Full grids | `tab:full-grid-sst2`, `tab:full-grid-squad` | `main/grid.yaml`, `references/nonprivate.yaml`, `sweeps/reference_learning_rate.yaml`, `references/zero_shot.yaml` |
| Per-seed results | `tab:per-seed-all` | `main/grid.yaml` |
| Unanimity per pool | `tab:unanimity-pools` | `main/grid.yaml` |
| Membership inference | `tab:mia`, `tab:mia-all` | `main/grid.yaml` |
| Learning-rate sweeps | `tab:lr-sweeps`, `tab:zpl-lr-sweep` | `sweeps/learning_rate.yaml` |
| Reference learning-rate sweeps, T=1000 and T=4000 | `tab:ref-lr-sweeps` | `sweeps/reference_learning_rate.yaml` |
| MI-budget sweep | `tab:plateau`, `tab:LoRA-plateau-adaptive`, `fig:plateau` | `sweeps/mi_budget.yaml` |
| PACZero-AGC_0 step-budget sweep, T=1500 seeds, PACZero-BAC_0 at T=4000 | `tab:zpl-ft-ladder`, `app:zpl-canonical`, `app:protocol-details` | `sweeps/step_budget.yaml` |
| Unanimity against universe size | `tab:n-sweep` | `sweeps/universe_size.yaml` |
| Clipping-threshold ablation | `tab:clip-ablation`, `tab:clip-ablation-ft`, `tab:clip-ablation-squad-lora` | `ablations/clipping.yaml` |
| LoRA rank ablation | `tab:rank-ablation` | `ablations/rank.yaml` |
| K-aggregation ablation | `tab:k-ablation-all` | `ablations/k.yaml`, with the K=1 column from `main/grid.yaml` |
| Mechanism decomposition | `tab:ablation` | `ablations/mechanism.yaml`, with the n=3 row from `main/grid.yaml` |
| In-house DP baseline reproduction and DP-AggZO parity | `tab:dp-cliff` | `dp/baselines.yaml` |
| Cost profile and cost sweeps | `tab:cost-profile`, `tab:cost-sweeps` | `cost/timing.yaml` |

The non-private reference rows come from `references/nonprivate.yaml` and, for settings without their own reference run, from the dev-selected arm in `sweeps/reference_learning_rate.yaml`. The T=4000 reference values of the protocol appendix are the `*_ref_*_T4000_s0` arms of `sweeps/reference_learning_rate.yaml`.

The cost tables time the runs of `cost/timing.yaml`, each alone on one exclusive H100. Seconds per step come from the job wall clock and from the training progress bar. Peak memory is the maximum of `nvidia-smi` sampled once per second.
