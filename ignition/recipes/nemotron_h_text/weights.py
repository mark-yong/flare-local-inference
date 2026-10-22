"""Nemotron-H text checkpoint transfers preserving each declared weight format."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader

from .config import ModelConfig


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: object
    layers: tuple[dict, ...]
    allocated_bytes: int
    draft: object = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None):
        before = torch.cuda.memory_allocated(device)
        with NemotronTextWeightLoader(manifest, device, shard=shard, progress=progress) as loader:
            config = loader.config
            embedding, norm = loader.load_tensors((loader.prefix + ".embeddings.weight",
                                                   loader.prefix + ".norm_f.weight"))
            if embedding.dtype != torch.bfloat16 or embedding.shape != (config.vocabulary, config.hidden):
                raise ValueError("Nemotron-H text requires a BF16 embedding matching the configured vocabulary")
            head = loader.load_linear("lm_head")
            layers = tuple(loader.load_layer(index) for index in range(len(config.layers)))
        return cls(config, shard, embedding, norm, head, layers,
                   torch.cuda.memory_allocated(device) - before)


class NemotronTextWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None):
        super().__init__(manifest, device, progress=progress)
        if "language_model.backbone.embeddings.weight" in manifest.tensors:
            raise ValueError("an omni-wrapped checkpoint belongs to the nemotron_h recipe")
        self.prefix = "backbone"
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        self.config.validate_shard(shard)

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if name.startswith((self.prefix + ".", "lm_head."))}

    def load_layer(self, index):
        c = self.config.text
        prefix = f"{self.prefix}.layers.{index}"
        mixer = prefix + ".mixer"
        result = dict(norm=self.load_tensor(prefix + ".norm.weight"))
        kind = self.config.layers[index]
        if kind == "mamba":
            a, d, bias, convolution, conv_bias, norm = self.load_tensors(tuple(mixer + "." + suffix
                for suffix in ("A_log", "D", "dt_bias", "conv1d.weight", "conv1d.bias", "norm.weight")))
            heads, channels = c["mamba_num_heads"], self.config.convolution_width
            if (a.shape != (heads,) or d.shape != (heads,) or bias.shape != (heads,)
                    or convolution.shape != (channels, 1, c["conv_kernel"])
                    or conv_bias.shape != (channels,) or norm.shape != (self.config.mamba_width,)):
                raise ValueError(f"{mixer}: Mamba parameter geometry differs from configuration")
            result.update(input=self.load_linear(mixer + ".in_proj"),
                output=self.load_linear(mixer + ".out_proj"), A=-a.float().exp(), D=d,
                dt_bias=bias, convolution=convolution.squeeze(1), convolution_bias=conv_bias, output_norm=norm)
        elif kind == "attention":
            q, k, v = self.load_tensors(tuple(mixer + "." + name for name in
                ("q_proj.weight", "k_proj.weight", "v_proj.weight")))
            hidden, dim = self.config.hidden, c["head_dim"]
            heads, groups = c["num_attention_heads"], c["num_key_value_heads"]
            if (q.shape != (heads * dim, hidden) or k.shape != (groups * dim, hidden)
                    or v.shape != (groups * dim, hidden)
                    or any(t.dtype != torch.bfloat16 for t in (q, k, v))):
                raise ValueError(f"{mixer}: attention projection geometry differs from configuration")
            output = self.load_tensor(mixer + ".o_proj.weight")
            if output.shape != (hidden, heads * dim) or output.dtype != torch.bfloat16:
                raise ValueError(f"{mixer}: attention output geometry differs from configuration")
            result.update(qkv=torch.cat((q, k, v)), output=output)
        else:
            router, correction = self.load_tensors((mixer + ".gate.weight",
                                                    mixer + ".gate.e_score_correction_bias"))
            count = c["n_routed_experts"]
            if router.shape != (count, self.config.hidden) or correction.shape != (count,):
                raise ValueError(f"{mixer}: router geometry differs from configuration")
            result.update(router=router, correction=correction,
                experts=self.load_fp4_experts(
                    tuple(f"{mixer}.experts.{i}" for i in range(count)),
                    hidden_size=self.config.hidden, intermediate_size=c["moe_intermediate_size"],
                    format="nvfp4", activation_mode="a16", activation="relu2"),
                shared_up=self.load_linear(mixer + ".shared_experts.up_proj"),
                shared_down=self.load_linear(mixer + ".shared_experts.down_proj"))
        return result
