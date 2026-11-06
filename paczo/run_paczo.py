import logging
import os
import random
from dataclasses import dataclass
from typing import Literal

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)

from paczo.pac_trainer import PACTrainer
from paczo.pac_utils import SubsetAwareDataset, init_private_rng
from src.metrics import calculate_metric
from src.tasks import get_task
from src.utils import (
    DataCollatorWithPaddingAndNesting,
    PACDataCollator,
    Prediction,
    count_time,
    encode_prompt,
    forward_wrap_with_option_len_dpzero,
    tokenizer_prepends_bos,
    write_metrics_to_file,
)


@dataclass
class PACZeroArguments(TrainingArguments):
    task_name: str = "SST2"
    num_train: int = 0
    num_dev: int = None
    num_eval: int = None
    train_set_seed: int = None
    result_file: str = None

    model_name: str = "facebook/opt-125m"
    load_float16: bool = False
    max_length: int = 2048

    trainer: Literal["none", "zo"] = "none"
    train_as_classification: bool = False

    zo_eps: float = 1e-3

    lora: bool = False
    lora_alpha: int = 16
    lora_r: int = 8

    sampling: bool = False
    temperature: float = 1.0
    num_beams: int = 1
    top_k: int = None
    top_p: float = 0.95
    max_new_tokens: int = 50
    eos_token: str = "\n"

    pac_m: int = 128
    pac_mi: float = 0.33  # nats
    pac_clip: float = 25.0
    pac_k: int = 1
    pac_adaptive_mi: bool = False  # beta_t = (budget left) / (steps left) instead of budget / T
    pac_zero_budget: Literal["coin", "majority"] = None
    pac_channel: Literal["gaussian", "binary"] = "gaussian"
    pac_diag_dir: str = ""
    pac_sdpa_attention: bool = False
    pac_dump_membership: str = ""

    no_privacy: bool = False
    ablation: Literal["none", "quant_full", "raw_half", "quant_half", "random_sign"] = "none"
    pac_zo_third_pass: bool = True  # off: the post-update forward that only feeds the logged train_loss is skipped
    pac_sort_by_len: bool = False
    pac_trim_pad: bool = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ns = init_private_rng(os.environ.get("PAC_NOISE_SEED"))
    print(f"PAC_NOISE_SEED={ns} (private stream; not derivable from seed={seed})", flush=True)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        with count_time(
            "Loading model with FP%d" % (16 if self.args.load_float16 else 32)
        ):
            free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
            config = AutoConfig.from_pretrained(self.args.model_name)
            torch_dtype = torch.float16 if self.args.load_float16 else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                config=config,
                attn_implementation="sdpa" if self.args.pac_sdpa_attention else "eager",
                device_map="auto",
                torch_dtype=torch_dtype,
                max_memory={
                    i: f"{free_in_GB - 5}GB"
                    for i in range(torch.cuda.device_count())
                },
            )
            model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)
        logger.info(
            "tokenizer: %s (fast=%s) for %s",
            type(tokenizer).__name__, tokenizer.is_fast, self.args.model_name,
        )

        model_type = model.config.model_type
        if model_type == "opt":
            tokenizer.bos_token_id = 0
        if tokenizer.pad_token_id is None:
            logger.warning(
                "%s has no pad_token_id; using eos_token_id=%s as pad",
                self.args.model_name, tokenizer.eos_token_id,
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # src/utils.py masks padding by config.pad_token_id, which newer families leave None.
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(
            "special tokens: model_type=%s bos=%s eos=%s pad=%s config_pad=%s "
            "prepends_bos=%s",
            model_type, tokenizer.bos_token_id, tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            model.config.pad_token_id,
            tokenizer_prepends_bos(tokenizer),
        )

        if self.args.lora:
            from src.lora import LoRA

            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha)

        return model, tokenizer

    def forward(self, input_ids, option_len=None, generation=False):
        input_ids = torch.tensor([input_ids]).to(self.model.device)
        if generation:
            args = self.args
            outputs = self.model.generate(
                input_ids,
                do_sample=args.sampling,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=min(
                    args.max_new_tokens, args.max_length - input_ids.size(1)
                ),
                num_return_sequences=1,
                eos_token_id=[
                    self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                    self.tokenizer.eos_token_id,
                ],
            )
            output_text = self.tokenizer.decode(
                outputs[0][input_ids.size(1) :], skip_special_tokens=True
            ).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[
                torch.arange(len(labels)).to(labels.device), labels
            ]
            selected_log_probs = selected_log_probs.cpu().detach()
            return selected_log_probs[-option_len:]

    def one_step_pred(self, eval_sample, verbose=False):
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")

        encoded_candidates, option_lens = encode_prompt(
            self.task.get_template(),
            eval_sample,
            self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens,
        )

        if self.task.generation:
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}")
            return Prediction(
                correct_candidate=eval_sample.correct_candidate,
                predicted_candidate=output_text,
            )
        else:
            scores = []
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(
                    encoded_candidate, option_len=option_lens[candidate_id]
                )
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info(
                            "=== Candidate %d (without context)===" % candidate_id
                        )
                        logger.info(
                            self.tokenizer.decode(encoded_candidate).split(
                                self.task.train_sep
                            )[-1]
                        )
                    logger.info(
                        f"Log probabilities of the option tokens: {selected_log_probs}"
                    )

                scores.append(selected_log_probs.mean().item())

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(
                correct_candidate=correct_candidate_id,
                predicted_candidate=int(np.argmax(scores)),
            )

    def evaluate(self, eval_samples):
        logger.info(f"Evaluating {len(eval_samples)} samples")
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(self.one_step_pred(eval_sample, verbose=(eval_id < 3)))
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def train(self, train_samples, eval_samples):
        self.tokenizer.padding_side = "left"
        logger.info("padding_side set to %s for training", self.tokenizer.padding_side)

        def _convert(samples):
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task.get_template(),
                    sample,
                    self.tokenizer,
                    max_length=self.args.max_length,
                    generation=self.task.generation,
                    generation_with_gold=True,
                    max_new_tokens=self.args.max_new_tokens,
                )
                if self.task.generation:
                    correct_candidate_id = 0
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)

                if self.args.train_as_classification:
                    data.append(
                        [
                            {
                                "input_ids": encoded_candidates[_i],
                                "labels": correct_candidate_id,
                                "option_len": option_lens[_i],
                                "num_options": len(sample.candidates),
                            }
                            for _i in range(len(encoded_candidates))
                        ]
                    )
                else:
                    data.append(
                        {
                            "input_ids": encoded_candidates[correct_candidate_id],
                            "labels": encoded_candidates[correct_candidate_id],
                            "option_len": option_lens[correct_candidate_id],
                        }
                    )
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = _convert(train_samples)
            eval_dataset = _convert(eval_samples)

        train_dataset = SubsetAwareDataset(train_dataset, m=self.args.pac_m)

        self.model.original_forward = self.model.forward
        self.model.forward = forward_wrap_with_option_len_dpzero.__get__(
            self.model, type(self.model)
        )

        if self.args.train_as_classification:
            inner_collator = DataCollatorWithPaddingAndNesting(
                self.tokenizer, pad_to_multiple_of=8
            )
        else:
            inner_collator = DataCollatorForTokenClassification(
                self.tokenizer, pad_to_multiple_of=8
            )

        collator = PACDataCollator(inner_collator)

        trainer = PACTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
        )

        trainer.train()
        self.model = trainer.model
        self.model.forward = self.model.original_forward


def main(args):
    set_seed(args.seed)
    task = get_task(args.task_name)
    train_samples = task.sample_train_set(
        num_train=args.num_train, num_dev=args.num_dev, seed=args.train_set_seed
    )
    framework = Framework(args, task)
    eval_samples = task.sample_subset(data_split="valid", seed=args.train_set_seed, num=args.num_eval)

    if args.trainer == "none":
        metrics = framework.evaluate(eval_samples)
    else:
        dev_samples = train_samples[-args.num_dev :]
        framework.train(train_samples[: -args.num_dev], dev_samples)
        metrics = framework.evaluate(eval_samples)
        for m, v in framework.evaluate(dev_samples).items():
            metrics["dev_" + m] = v

    logger.info("===== Train set %d =====" % args.train_set_seed)
    logger.info(metrics)
    write_metrics_to_file(metrics, args.result_file)
