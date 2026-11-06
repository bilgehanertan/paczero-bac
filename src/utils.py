import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


logger = logging.getLogger(__name__)


def tokenizer_prepends_bos(tokenizer):
    """Probes `encode`: `add_bos_token` cannot tell, since Llama-3 tokenizers lack the attribute and
    prepend BOS while Qwen2 tokenizers lack it and do not."""
    cached = getattr(tokenizer, "_pac_prepends_bos", None)
    if cached is not None:
        return cached
    bos = tokenizer.bos_token_id
    ids = tokenizer.encode("x") if bos is not None else []
    tokenizer._pac_prepends_bos = bool(ids) and ids[0] == bos
    return tokenizer._pac_prepends_bos

# CausalLM classes whose forward upcasts the head output to fp32 before the loss;
# the scored-positions path must match the full forward per family.
_PAC_UPCAST_LOGITS = {"LlamaForCausalLM", "Qwen2ForCausalLM"}


def _scored_positions_loss(self, input_ids, option_len, **kwargs):
    base_out = self.model(input_ids=input_ids, **kwargs)
    hidden = base_out[0]

    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100
    for _i, _len in enumerate(option_len):
        shift_labels[_i, :-_len] = -100

    mask = shift_labels != -100
    rows, cols = mask.nonzero(as_tuple=True)
    sel_hidden = hidden[:, :-1, :][rows, cols]
    sel_logits = self.lm_head(sel_hidden)
    if type(self).__name__ in _PAC_UPCAST_LOGITS:
        sel_logits = sel_logits.float()

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    ce = loss_fct(sel_logits, shift_labels[rows, cols])
    total = torch.zeros(shift_labels.size(0), dtype=ce.dtype, device=ce.device)
    total.index_add_(0, rows, ce)
    loss = total / mask.sum(dim=1)

    return (loss, None)


def forward_wrap_with_option_len_dpzero(self, input_ids=None, labels=None, option_len=None, num_options=None, **kwargs):
    if num_options is None:
        return _scored_positions_loss(self, input_ids, option_len, **kwargs)

    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    # input_ids, not labels: labels hold the candidate index.
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100
    for _i, _len in enumerate(option_len):
        shift_labels[_i, :-_len] = -100

    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    shift_labels[~mask] = 0
    selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1)

    num_options = num_options[0]
    selected_log_probs = selected_log_probs.view(-1, num_options)
    labels = labels.view(-1, num_options)[:, 0]
    loss = CrossEntropyLoss(ignore_index=-100, reduction='none')(selected_log_probs, labels)

    return (loss, logits) + outputs[1:]


def encode_prompt(template, eval_sample, tokenizer, max_length, generation=False, generation_with_gold=False, max_new_tokens=None):
    unverbalized_eval_prompt = template.encode(eval_sample).strip(' ')
    if not generation:
        verbalized_eval_prompts = [template.verbalize(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]
        final_prompts = [eval_prompt.lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts]
    elif generation_with_gold:
        verbalized_eval_prompts = [template.verbalize(eval_sample, eval_sample.correct_candidate)]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]
        final_prompts = [eval_prompt.lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts]
    else:
        option_lens = [0]
        final_prompts = [unverbalized_eval_prompt.lstrip().strip(' ')]

    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]

    if generation and max_new_tokens is not None:
        max_length = max_length - max_new_tokens

    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warn("Exceed max length")
    if tokenizer_prepends_bos(tokenizer):
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]

    return encodings, option_lens


@dataclass
class DataCollatorWithPaddingAndNesting:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [ff for f in features for ff in f]
        return self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]


@contextlib.contextmanager
def count_time(name):
    logger.info("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        logger.info("Done with %.2fs" % (time.time() - start_time))


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def write_metrics_to_file(metrics, output):
    json.dump(metrics, open(output, "w"), indent=4)


def _without_membership(feature):
    return {k: v for k, v in feature.items() if k != 'membership_vector'}


class PACDataCollator:
    def __init__(self, inner_collator):
        self.inner_collator = inner_collator

    def __call__(self, features):
        membership_vectors = []
        clean_features = []
        for f in features:
            first = f[0] if isinstance(f, list) else f
            if 'membership_vector' in first:
                membership_vectors.append(first['membership_vector'])
            if isinstance(f, list):
                clean_features.append([_without_membership(d) for d in f])
            else:
                clean_features.append(_without_membership(f))

        batch = self.inner_collator(clean_features)
        if membership_vectors:
            batch['membership_vector'] = torch.stack(membership_vectors)
        return batch
