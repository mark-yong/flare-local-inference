"""Nemotron-H text (NemotronHForCausalLM) heterogeneous block geometry and execution semantics.

The Nemotron 3.5 text checkpoints (Lightning) declare their hybrid layout and MoE
geometry in top-level fields and carry an in-checkpoint MTP module that this
recipe does not prepare.
"""

from dataclasses import dataclass
import json
from pathlib import Path

PAGE_TOKENS = 128


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    eos_token: int
    draft: object = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw):
        if (raw.get("model_type") != "nemotron_h"
                or raw.get("architectures") != ["NemotronHForCausalLM"]):
            raise ValueError("Nemotron-H text requires NemotronHForCausalLM")
        text = raw
        defaults = dict(use_qk_norm=False, scoring_func="sigmoid", time_step_limit=(0.0, float("inf")))
        required = dict(mamba_hidden_act="silu", mlp_hidden_act="relu2",
            use_bias=False, mlp_bias=False, attention_bias=False, use_qk_norm=False,
            use_conv_bias=True, conv_kernel=4, n_group=1, topk_group=1,
            norm_topk_prob=True, scoring_func="sigmoid", residual_in_fp32=False,
            tie_word_embeddings=False, sliding_window=None, time_step_limit=(0.0, float("inf")))
        for key, expected in required.items():
            if text.get(key, defaults.get(key)) != expected:
                raise ValueError(f"unsupported Nemotron-H text {key}: {text.get(key)!r}")
        for key in ("hidden_size", "vocab_size", "mamba_num_heads", "mamba_head_dim", "ssm_state_size",
                    "n_groups", "num_attention_heads", "num_key_value_heads", "head_dim",
                    "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size",
                    "moe_shared_expert_intermediate_size",
                    "max_position_embeddings"):
            if type(text.get(key)) is not int or text[key] <= 0:
                raise ValueError(f"Nemotron-H text requires positive {key}")
        scale = text.get("routed_scaling_factor")
        if type(scale) not in (int, float) or scale <= 0:
            raise ValueError(f"unsupported Nemotron-H text routed_scaling_factor: {scale!r}")
        if text["n_shared_experts"] != 1:
            raise ValueError(f"unsupported Nemotron-H text n_shared_experts: {text['n_shared_experts']!r}")
        if (text["mamba_head_dim"] not in (64, 128) or text["ssm_state_size"] not in (64, 128)
                or text["mamba_num_heads"] % text["n_groups"]
                or text["num_attention_heads"] % text["num_key_value_heads"]):
            raise ValueError("unsupported Nemotron-H text attention or state geometry")
        if text["moe_intermediate_size"] % 16 or text["moe_shared_expert_intermediate_size"] % 16:
            raise ValueError("Nemotron-H text MoE widths must contain complete NVFP4 blocks")
        if "block_configs" in text or "vision_config" in raw:
            raise ValueError("Nemotron-H text does not prepare block configurations or vision encoders")
        if text.get("layers_block_type") is None or len(text["layers_block_type"]) != text.get("num_hidden_layers"):
            raise ValueError("Nemotron-H text requires a layer sequence matching num_hidden_layers")
        for kind in text["layers_block_type"]:
            if kind not in ("mamba", "attention", "moe"):
                raise ValueError(f"unsupported Nemotron-H text block: {kind}")
        return cls(text, raw["eos_token_id"])

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def layers(self):
        return tuple(self.text["layers_block_type"])

    @property
    def mamba_width(self):
        return self.text["mamba_num_heads"] * self.text["mamba_head_dim"]

    @property
    def convolution_width(self):
        return self.mamba_width + 2 * self.text["n_groups"] * self.text["ssm_state_size"]

    def validate_shard(self, shard):
        if shard.world_size != 1:
            raise ValueError("Nemotron-H text preparation requires one GPU")
