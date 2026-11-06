import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_callback import ExportableState, TrainerState
from transformers.trainer_utils import TrainOutput, seed_worker, speed_metrics
from transformers.utils import logging

from paczo.pac_utils import (
    channel_for_binary_mi,
    compute_subset_ghats,
    private_randint,
    private_randn,
    private_random,
    sample_binary_channel,
    sigma_for_binary_mi,
    update_p_binary,
    update_p_channel,
)

logger = logging.get_logger(__name__)

MICRO_BATCH_SIZE = 64
CHANNEL_RELEASES = ("channel_flip", "channel_identity", "majority")
MEAN_OVER_DIRECTIONS = (
    "signal_raw", "batch_mean", "var_subset_raw", "q_plus", "sigma_used", "ch_alpha", "ch_beta",
    "flip_distortion_D", "pos_sign_count", "s_secret", "released_bit", "projected_grad",
    "pre_clip_mean", "pre_clip_max", "clip_fired_frac",
)


@dataclass
class Release:
    bit: float
    grad: float  # the coefficient of z in the update
    mi: float  # MI charged, nats
    kind: str
    sigma: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    D: float = 0.0


def subset_estimate(ghat, membership, clip, step):
    sample_abs = np.abs(ghat)
    clip_scale = np.minimum(1.0, clip / np.maximum(sample_abs, 1e-12))
    ghat_clipped = ghat * clip_scale
    bad = int((~np.isfinite(ghat)).sum())
    if bad:
        raise RuntimeError(
            "PACZero step %s: %d/%d per-sample estimates are not finite. Refusing to "
            "sign-quantize: NaN would be reported as unanimity at zero MI." % (step, bad, ghat.size)
        )
    m_ghats = compute_subset_ghats(ghat_clipped, membership)
    signs = np.where(m_ghats >= 0, 1.0, -1.0).astype(np.float64)
    return {
        "m_ghats": m_ghats,
        "signs": signs,
        "batch_mean": float(ghat_clipped.mean()),
        "pre_clip_mean": float(sample_abs.mean()),
        "pre_clip_max": float(sample_abs.max()),
        "pre_clip_nonzero": int((sample_abs > 0).sum()),
        "clip_fired_frac": float((sample_abs > clip).mean()),
        "batch_size": len(ghat),
    }


def release_nonprivate(ablation, est, p, budget):
    s = est["s_secret"]
    if ablation == "none":
        grad = est["batch_mean"]
    elif ablation == "quant_full":
        grad = 1.0 if est["batch_mean"] >= 0 else -1.0
    elif ablation == "raw_half":
        grad = float(est["m_ghats"][est["secret"]])
    elif ablation == "quant_half":
        grad = s
    else:
        grad = 1.0 if np.random.random() < 0.5 else -1.0
    # The posterior a gaussian release would have produced, for the diagnostics only.
    if not est["unanimous"]:
        sigma = sigma_for_binary_mi(est["q_plus"], budget)
        p = update_p_binary(p, est["signs"], s + private_randn() * sigma, sigma)
    return Release(s, grad, 0.0, f"ablation_{ablation}"), p


def release_zero_budget(mode, est, p_live):
    s = est["s_secret"]
    if est["unanimous_true"]:
        return Release(s, s, 0.0, "unanimous"), p_live
    if mode == "majority":
        alpha, beta, D = channel_for_binary_mi(est["q_plus"], 0.0)
        if alpha == 0.5 and beta == 0.5:
            # An exact tie must not reach the sampler, which would mask s_secret with a coin.
            # The majority is constant across candidates, logged as the (1, 0) channel.
            bit = est["majority_bit"]
            alpha, beta = 1.0, 0.0
        else:
            bit = sample_binary_channel(s, alpha, beta)
        p_live = update_p_channel(p_live, est["signs"], bit, alpha, beta)
        return Release(bit, bit, 0.0, "majority", alpha=alpha, beta=beta, D=D), p_live
    bit = 1.0 if private_random() < 0.5 else -1.0
    return Release(bit, bit, 0.0, "coin"), p_live


def release_mi(channel, est, p, budget, h):
    s, signs = est["s_secret"], est["signs"]
    if est["unanimous"]:
        return Release(s, s, 0.0, "unanimous"), p
    if channel == "binary":
        alpha, beta, D = channel_for_binary_mi(est["q_plus"], budget)
        if D <= 0.0:
            # budget >= h(q+): the identity channel, charged h(q+); alpha = beta = 0 is its exact update.
            p = update_p_channel(p, signs, s, 0.0, 0.0)
            return Release(s, s, float(h), "channel_identity", alpha=alpha, beta=beta, D=D), p
        bit = sample_binary_channel(s, alpha, beta)
        p = update_p_channel(p, signs, bit, alpha, beta)
        return Release(bit, bit, float(budget), "channel_flip", alpha=alpha, beta=beta, D=D), p
    sigma = sigma_for_binary_mi(est["q_plus"], budget)
    released_real = s + private_randn() * sigma
    bit = 1.0 if released_real >= 0 else -1.0
    p = update_p_binary(p, signs, released_real, sigma)
    return Release(bit, bit, float(budget), "noisy_gaussian", sigma=sigma), p


def sequential_sum(values):
    """Left-to-right float sum. sum() compensates rounding on Python 3.12+, which moves the last bits."""
    total = 0.0
    for v in values:
        total += v
    return total


def combine_directions(rows, step_budget):
    out = dict(rows[-1])
    for key in MEAN_OVER_DIRECTIONS:
        out[key] = float(np.mean([r[key] for r in rows]))
    out["mi_step"] = float(sequential_sum(r["mi_step"] for r in rows))
    out["per_step_mi_budget"] = float(step_budget)
    out["per_step_mi_capped"] = any(r["per_step_mi_capped"] for r in rows)
    out["unanimous"] = all(r["unanimous"] for r in rows)
    out["unanimous_true"] = all(r["unanimous_true"] for r in rows)
    out["release_type"] = Counter(r["release_type"] for r in rows).most_common(1)[0][0]
    out["pre_clip_nonzero"] = int(np.min([r["pre_clip_nonzero"] for r in rows]))
    return out


def _z(param):
    return torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)


class PACTrainer(Trainer):

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=len(self.train_dataset),
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
        )


    def _perturb(self, params, seed, scale):
        torch.manual_seed(seed)
        for _, param in params:
            param.data = param.data + scale * _z(param)

    def _two_point_estimate(self, model, inputs, params, seed):
        eps = self.args.zo_eps
        self._perturb(params, seed, eps)
        loss_plus = self._per_sample_losses(model, inputs)
        self._perturb(params, seed, -2.0 * eps)
        loss_minus = self._per_sample_losses(model, inputs)
        ghat = (loss_plus.float() - loss_minus.float()) / (2.0 * eps)
        ghat = ghat.detach().cpu().numpy().astype(np.float64)
        self._perturb(params, seed, eps)
        two_pass_loss = None if self.args.pac_zo_third_pass else 0.5 * (loss_plus.mean() + loss_minus.mean())
        return ghat, two_pass_loss

    def _update(self, params, seeds, grads, lr):
        if len(seeds) == 1:
            # Streams z per parameter; the buffered path below holds an extra copy of every trainable parameter.
            torch.manual_seed(seeds[0])
            for _, param in params:
                param.data = param.data - lr * (grads[0] * _z(param))
            return
        inv_k = 1.0 / len(seeds)
        buffers = {name: torch.zeros_like(param.data) for name, param in params}
        for seed, grad in zip(seeds, grads):
            torch.manual_seed(seed)
            coef = grad * inv_k
            for name, param in params:
                buffers[name] = buffers[name] + coef * _z(param)
        for name, param in params:
            param.data = param.data - lr * buffers[name]

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self._train_batch_size = batch_size
        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        max_steps = args.max_steps
        num_train_epochs = max_steps
        num_examples = self.num_examples(train_dataloader)
        num_train_samples = max_steps * total_train_batch_size

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.train_batch_size = self._train_batch_size
        for name in ("logging_steps", "eval_steps", "save_steps"):
            setattr(self.state, name, getattr(args, name))

        model = self._wrap_model(self.model_wrapped)
        if model is not self.model:
            self.model_wrapped = model

        no_privacy, zero_budget = args.no_privacy, args.pac_zero_budget
        pac_m, pac_mi, pac_k = args.pac_m, args.pac_mi, args.pac_k
        secret = private_randint(pac_m)

        p = np.ones(pac_m, dtype=np.float64) / pac_m
        p_live = p.copy()
        uniform_budget = pac_mi / max_steps
        cum_mi = 0.0
        n_unanimous_steps = 0
        n_disagreement_steps = 0

        diag_dir = args.pac_diag_dir
        os.makedirs(diag_dir, exist_ok=True)
        diag_file = os.path.join(diag_dir, "diagnostics.jsonl")
        open(diag_file, "w").close()
        os.makedirs(args.output_dir, exist_ok=True)

        n_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print("=" * 60)
        if no_privacy:
            print("  MODE: non-private reference, ablation=%s" % args.ablation)
        elif zero_budget:
            print("  MODE: PACZero-%s" % {"coin": "AGC_0", "majority": "BAC_0"}[zero_budget])
        else:
            variant = {"gaussian": "AGC", "binary": "BAC"}[args.pac_channel]
            alloc = "adaptive (budget left / steps left)" if args.pac_adaptive_mi else "uniform (MI / T)"
            print(f"  MODE: PACZero-{variant}, allocation {alloc}")
        print(f"  K={pac_k}, M={pac_m}, total_MI={pac_mi}, per_step_MI={uniform_budget:.8f}")
        print(f"  secret_id={secret}, clip={args.pac_clip}")
        print(f"  max_steps={max_steps}, trainable_params={n_trainable:,}")
        print("=" * 60)

        logger.info("***** Running PACZero training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Trainable parameters = {n_trainable:,}")

        self.state.epoch = 0
        start_time = time.time()

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for epoch in range(max_steps):
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            for inputs in train_dataloader:
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                step = self.state.global_step
                membership = inputs.pop("membership_vector").numpy().astype(np.bool_)
                if args.pac_dump_membership and step == 0:
                    np.savez(args.pac_dump_membership, membership=membership)
                    print(f"dumped membership {membership.shape} to {args.pac_dump_membership}; exiting.", flush=True)
                    raise SystemExit(0)

                params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

                if args.pac_adaptive_mi:
                    step_budget = max(pac_mi - cum_mi, 0.0) / (max_steps - step)
                else:
                    step_budget = uniform_budget

                seeds, releases, rows = [], [], []
                for _ in range(pac_k):
                    seed = int(np.random.randint(1000000000))
                    ghat, two_pass_loss = self._two_point_estimate(model, inputs, params, seed)
                    est = subset_estimate(ghat, membership, args.pac_clip, step)
                    signs = est["signs"]
                    q_plus = float(np.sum(p * (signs > 0).astype(np.float64)))
                    pos_count = int((signs > 0).sum())
                    est.update(
                        secret=secret,
                        s_secret=float(signs[secret]),
                        q_plus=q_plus,
                        unanimous=(q_plus <= 1e-12) or (q_plus >= 1.0 - 1e-12),
                        unanimous_true=(pos_count == 0) or (pos_count == pac_m),
                        majority_bit=1.0 if (2 * pos_count - pac_m) >= 0 else -1.0,
                    )
                    entropy = float(-np.sum(p * np.log(p + 1e-300)))

                    budget = step_budget / pac_k
                    capped = False
                    h = 0.0
                    if 0.0 < q_plus < 1.0:
                        h = float(-q_plus * np.log(q_plus) - (1.0 - q_plus) * np.log(1.0 - q_plus))
                        # The binary channel covers budget >= h(q+) with its identity branch.
                        if args.pac_channel == "gaussian" and budget > 0.999 * h:
                            budget = 0.999 * h
                            capped = True

                    if no_privacy:
                        rel, p = release_nonprivate(args.ablation, est, p, budget)
                    elif zero_budget:
                        rel, p_live = release_zero_budget(zero_budget, est, p_live)
                    else:
                        rel, p = release_mi(args.pac_channel, est, p, budget, h)

                    seeds.append(seed)
                    releases.append(rel)
                    rows.append({
                        "signs": signs,
                        "m_ghats": est["m_ghats"],
                        "signal_raw": float(np.abs(est["m_ghats"][secret])),
                        "batch_mean": est["batch_mean"],
                        "var_subset_raw": float(np.var(est["m_ghats"])),
                        "q_plus": q_plus,
                        "unanimous": est["unanimous"],
                        "sigma_used": float(rel.sigma),
                        "mi_step": float(rel.mi),
                        "ch_alpha": float(rel.alpha),
                        "ch_beta": float(rel.beta),
                        "flip_distortion_D": float(rel.D),
                        "per_step_mi_budget": float(budget),
                        "per_step_mi_capped": capped,
                        "release_type": rel.kind,
                        "unanimous_true": est["unanimous_true"],
                        "pos_sign_count": pos_count,
                        "majority_bit": est["majority_bit"],
                        "majority_margin": abs(2 * pos_count - pac_m) / float(pac_m),
                        "s_secret": est["s_secret"],
                        "released_bit": float(rel.bit),
                        "entropy": entropy,
                        "pre_clip_mean": est["pre_clip_mean"],
                        "pre_clip_max": est["pre_clip_max"],
                        "pre_clip_nonzero": est["pre_clip_nonzero"],
                        "clip_fired_frac": est["clip_fired_frac"],
                        "batch_size": est["batch_size"],
                        "projected_grad": float(rel.grad),
                    })

                if not no_privacy:
                    step_mi = sequential_sum(r.mi for r in releases)
                    cum_mi += step_mi
                    if zero_budget:
                        disagreement = any(r.kind != "unanimous" for r in releases)
                    else:
                        disagreement = step_mi > 0.0
                    if disagreement:
                        n_disagreement_steps += 1
                    else:
                        n_unanimous_steps += 1

                lr = self._get_learning_rate()
                self._update(params, seeds, [r.grad for r in releases], lr)
                self.lr_scheduler.step()

                tr_loss_step = self._per_sample_losses(model, inputs).mean() if args.pac_zo_third_pass else two_pass_loss
                tr_loss += tr_loss_step
                torch.cuda.empty_cache()
                self.current_flos += float(self.floating_point_ops(inputs))

                summary = rows[0] if pac_k == 1 else combine_directions(rows, step_budget)
                self._write_diagnostics(diag_file, summary, secret, p, p_live, cum_mi, n_unanimous_steps,
                                        n_disagreement_steps, tr_loss_step, lr)

                if step % args.logging_steps == 0:
                    self.log({
                        "signal_raw": summary["signal_raw"],
                        "q_plus": summary["q_plus"],
                        "agreement_frac": max(summary["q_plus"], 1.0 - summary["q_plus"]),
                        "sigma_used": summary["sigma_used"],
                        "cum_mi": float(cum_mi),
                        "free_frac": n_unanimous_steps / max(n_unanimous_steps + n_disagreement_steps, 1),
                        "entropy_ratio": summary["entropy"] / float(np.log(pac_m)),
                        "clip_fired_frac": summary["clip_fired_frac"],
                    })

                self.state.global_step += 1
                self.state.epoch = epoch + 1
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(torch.tensor(0.0), None, model, trial, epoch, ignore_keys_for_eval)

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(torch.tensor(0.0), None, model, trial, epoch, ignore_keys_for_eval)
            if self.control.should_training_stop:
                break

        logger.info("\n\nTraining completed.\n\n")
        train_loss = float(tr_loss.item()) / max(self.state.global_step, 1)
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best-dev checkpoint {self.state.best_model_checkpoint} "
                f"({self.args.metric_for_best_model}={self.state.best_metric})"
            )
            self._load_best_model()

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _write_diagnostics(self, diag_file, s, secret, p, p_live, cum_mi, n_unanimous_steps, n_disagreement_steps,
                           tr_loss_step, lr):
        args = self.args
        sign_digest = hashlib.sha256(np.ascontiguousarray(s["signs"], dtype=np.float64).tobytes()).hexdigest()[:16]
        # The full MT19937 state: randn() caches a Box-Muller variate that keys and position do not show.
        npst = np.random.get_state()
        np_digest = hashlib.sha256(
            npst[1].tobytes() + np.int64(npst[2]).tobytes() + np.int64(npst[3]).tobytes()
            + np.float64(npst[4]).tobytes()
        ).hexdigest()[:16]
        if s["release_type"] in CHANNEL_RELEASES:
            snr = float("nan")
        else:
            snr = s["signal_raw"] / (s["sigma_used"] if s["sigma_used"] > 0 else 1e-12)
        row = {
            "subset_sign_digest": sign_digest,
            "min_abs_subset_mean": float(np.abs(s["m_ghats"]).min()),
            "np_rng_pos": int(npst[2]),
            "np_rng_has_gauss": int(npst[3]),
            "np_rng_digest": np_digest,
            "step": self.state.global_step,
            "signal_raw": s["signal_raw"],
            "batch_mean": s["batch_mean"],
            "var_subset_raw": s["var_subset_raw"],
            "q_plus": s["q_plus"],
            "agreement_frac": max(s["q_plus"], 1.0 - s["q_plus"]),
            "unanimous": bool(s["unanimous"]),
            "sigma_used": s["sigma_used"],
            "mi_step": s["mi_step"],
            "channel": args.pac_channel,
            "ch_alpha": s["ch_alpha"],
            "ch_beta": s["ch_beta"],
            "flip_distortion_D": s["flip_distortion_D"],
            "per_step_mi_budget": s["per_step_mi_budget"],
            "per_step_mi_capped": bool(s["per_step_mi_capped"]),
            "adaptive_mi": bool(args.pac_adaptive_mi),
            "zero_budget": args.pac_zero_budget,
            "release_type": s["release_type"],
            "unanimous_true": bool(s["unanimous_true"]),
            "pos_sign_count": s["pos_sign_count"],
            "pac_secret_id": int(secret),
            "majority_bit": float(s["majority_bit"]),
            "majority_margin": float(s["majority_margin"]),
            "mech_seed": int(args.seed),
            "p_true": float(p[secret]),
            "p_true_live": float(p_live[secret]),
            "p_live_max_dev": float(np.max(np.abs(p_live - 1.0 / args.pac_m))),
            "secret_rank": int((p > p[secret]).sum()),
            "cum_mi": float(cum_mi),
            "n_free_steps": n_unanimous_steps,
            "n_noisy_steps": n_disagreement_steps,
            "s_secret": s["s_secret"],
            "released_bit": s["released_bit"],
            "snr_per_step_raw": snr,
            "entropy": s["entropy"],
            "entropy_ratio": s["entropy"] / float(np.log(args.pac_m)),
            "pre_clip_mean": s["pre_clip_mean"],
            "pre_clip_max": s["pre_clip_max"],
            "pre_clip_nonzero": s["pre_clip_nonzero"],
            "clip_fired_frac": s["clip_fired_frac"],
            "batch_size": s["batch_size"],
            "projected_grad": s["projected_grad"],
            "train_loss": float(tr_loss_step.item()),
            "lr": lr,
            "pac_clip": float(args.pac_clip),
            "pac_m": args.pac_m,
            "mode": "diagnostic" if args.no_privacy else "private",
        }
        with open(diag_file, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def _per_sample_losses(self, model, inputs):
        model.eval()
        order = None
        if self.args.pac_sort_by_len:
            order = torch.argsort(inputs["attention_mask"].sum(dim=1), stable=True)
            inputs = {k: v[order] for k, v in inputs.items()}

        losses = []
        for start in range(0, inputs["input_ids"].size(0), MICRO_BATCH_SIZE):
            micro = {k: v[start:start + MICRO_BATCH_SIZE] for k, v in inputs.items()}
            if self.args.pac_trim_pad:
                # Padding is on the left, so columns dead for every row of the chunk drop out exactly.
                mask = micro["attention_mask"]
                live = int(mask.sum(dim=0).nonzero()[0].item())
                micro = {k: v[:, live:] if v.dim() == 2 and v.size(1) == mask.size(1) else v for k, v in micro.items()}
            with torch.inference_mode():
                micro = self._prepare_inputs(micro)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, micro)
            losses.append(loss.detach())
            del micro
            torch.cuda.empty_cache()

        out = torch.cat(losses, dim=0)
        if order is not None:
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel(), device=order.device)
            out = out[inverse]
        return out

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs[0]
        return (loss, outputs) if return_outputs else loss
