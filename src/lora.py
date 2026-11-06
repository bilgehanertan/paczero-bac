import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r, lora_alpha, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.scaling = lora_alpha / r
        self.weight.requires_grad = False
        # Draws weight and bias a second time before lora_A. The base weight is
        # overwritten afterwards, but the draws fix the RNG stream lora_A comes from.
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # nn.Linear.__init__ calls this before lora_A exists.
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)
        result += (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result


class LoRA:
    def __init__(self, model, r, alpha):
        for key, attn in model.named_modules():
            if not key.endswith("attn"):
                continue
            logger.info(f"Inject lora to: {key}")
            for proj_name in ("q_proj", "v_proj"):
                orig = getattr(attn, proj_name)
                has_bias = orig.bias is not None
                new_layer = LoRALinear(
                    orig.in_features, orig.out_features, r=r, lora_alpha=alpha, bias=has_bias,
                ).to(device=orig.weight.device, dtype=orig.weight.dtype)
                new_layer.weight.data = orig.weight.data
                if has_bias:
                    new_layer.bias.data = orig.bias.data
                setattr(attn, proj_name, new_layer)

        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
