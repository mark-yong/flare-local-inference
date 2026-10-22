"""Layer weight loading: geometry validation, qkv fusion, and mamba parameter wiring.

Fake sources are injected into the loader the same way the component weight
tests do; no checkpoint files or CUDA are involved. The ungated relu2 expert
branch is covered by tests/components/test_weight_formats.py.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.nemotron_h_text.config import ModelConfig
from ignition.recipes.nemotron_h_text.weights import NemotronTextWeightLoader


def make_loader(lightning_config, tmp_path, tensors):
    (tmp_path / "config.json").write_text(json.dumps(lightning_config))
    manifest = SimpleNamespace(snapshot=tmp_path, tensors={name: None for name in tensors})
    loader = NemotronTextWeightLoader(manifest, "cpu", shard=SimpleNamespace(world_size=1))
    loader._session = SimpleNamespace(flush=lambda: None)
    loader._copy = lambda destination, source: destination.copy_(source)
    loader._sources.update(tensors)
    return loader


def source(name, tensor):
    return {name: tensor}


@pytest.fixture()
def indexes(lightning_config):
    kinds = lightning_config["layers_block_type"]
    return SimpleNamespace(mamba=kinds.index("mamba"), attention=kinds.index("attention"),
                           moe=kinds.index("moe"))


def attention_sources(config, index):
    c, hidden = config.text, config.hidden
    heads, groups, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    q = torch.full((heads * dim, hidden), 1.0, dtype=torch.bfloat16)
    k = torch.full((groups * dim, hidden), 2.0, dtype=torch.bfloat16)
    v = torch.full((groups * dim, hidden), 3.0, dtype=torch.bfloat16)
    o = torch.full((hidden, heads * dim), 4.0, dtype=torch.bfloat16)
    norm = torch.ones(hidden, dtype=torch.bfloat16)
    prefix = f"backbone.layers.{index}.mixer"
    return {f"backbone.layers.{index}.norm.weight": norm, f"{prefix}.q_proj.weight": q,
            f"{prefix}.k_proj.weight": k, f"{prefix}.v_proj.weight": v, f"{prefix}.o_proj.weight": o}, (q, k, v, o)


def test_attention_layer_fuses_qkv_in_projection_order(lightning_config, tmp_path, indexes):
    config = ModelConfig.from_dict(lightning_config)
    c = config.text
    heads, groups, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    tensors, (q, k, v, o) = attention_sources(config, indexes.attention)
    loader = make_loader(lightning_config, tmp_path, tensors)
    layer = loader.load_layer(indexes.attention)
    assert layer["norm"].shape == (config.hidden,)
    assert layer["qkv"].shape == ((heads + 2 * groups) * dim, config.hidden)
    assert torch.equal(layer["qkv"][:heads * dim], q)
    assert torch.equal(layer["qkv"][heads * dim:(heads + groups) * dim], k)
    assert torch.equal(layer["qkv"][(heads + groups) * dim:], v)
    assert torch.equal(layer["output"], o)


def test_attention_layer_rejects_wrong_projection_dtype(lightning_config, tmp_path, indexes):
    config = ModelConfig.from_dict(lightning_config)
    tensors, (q, k, v, o) = attention_sources(config, indexes.attention)
    prefix = f"backbone.layers.{indexes.attention}.mixer"
    tensors[f"{prefix}.q_proj.weight"] = q.float()
    loader = make_loader(lightning_config, tmp_path, tensors)
    with pytest.raises(ValueError, match="attention projection geometry"):
        loader.load_layer(indexes.attention)


def test_attention_layer_rejects_wrong_output_shape(lightning_config, tmp_path, indexes):
    config = ModelConfig.from_dict(lightning_config)
    tensors, (q, k, v, o) = attention_sources(config, indexes.attention)
    prefix = f"backbone.layers.{indexes.attention}.mixer"
    tensors[f"{prefix}.o_proj.weight"] = o[:, :-1].contiguous()
    loader = make_loader(lightning_config, tmp_path, tensors)
    with pytest.raises(ValueError, match="output geometry"):
        loader.load_layer(indexes.attention)


def mamba_sources(config, index):
    c, channels = config.text, config.convolution_width
    heads, width = c["mamba_num_heads"], config.mamba_width
    prefix = f"backbone.layers.{index}.mixer"
    a_log = torch.full((heads,), 0.5)
    d = torch.full((heads,), 2.0)
    dt_bias = torch.full((heads,), 3.0)
    conv = torch.full((channels, 1, c["conv_kernel"]), 4.0, dtype=torch.bfloat16)
    conv_bias = torch.full((channels,), 5.0, dtype=torch.bfloat16)
    norm = torch.full((width,), 6.0, dtype=torch.bfloat16)
    tensors = {f"backbone.layers.{index}.norm.weight": norm, f"{prefix}.A_log": a_log,
               f"{prefix}.D": d, f"{prefix}.dt_bias": dt_bias, f"{prefix}.conv1d.weight": conv,
               f"{prefix}.conv1d.bias": conv_bias, f"{prefix}.norm.weight": norm}
    return tensors, (a_log, d, dt_bias, conv, conv_bias, norm)


def test_mamba_layer_wires_parameters_with_negated_exponent(lightning_config, tmp_path, indexes):
    config = ModelConfig.from_dict(lightning_config)
    tensors, (a_log, d, dt_bias, conv, conv_bias, norm) = mamba_sources(config, indexes.mamba)
    loader = make_loader(lightning_config, tmp_path, tensors)
    loader.load_linear = lambda name: ("linear", name)
    layer = loader.load_layer(indexes.mamba)
    assert torch.equal(layer["A"], -a_log.float().exp())
    assert torch.equal(layer["D"], d)
    assert torch.equal(layer["dt_bias"], dt_bias)
    assert layer["convolution"].shape == (config.convolution_width, config.text["conv_kernel"])
    assert torch.equal(layer["convolution"], conv.squeeze(1))
    assert torch.equal(layer["convolution_bias"], conv_bias)
    assert torch.equal(layer["output_norm"], norm)
    assert layer["input"] == ("linear", f"backbone.layers.{indexes.mamba}.mixer.in_proj")
    assert layer["output"] == ("linear", f"backbone.layers.{indexes.mamba}.mixer.out_proj")


def test_mamba_layer_rejects_wrong_parameter_geometry(lightning_config, tmp_path, indexes):
    config = ModelConfig.from_dict(lightning_config)
    tensors, _ = mamba_sources(config, indexes.mamba)
    tensors[f"backbone.layers.{indexes.mamba}.mixer.A_log"] = torch.full((4,), 0.5)
    loader = make_loader(lightning_config, tmp_path, tensors)
    with pytest.raises(ValueError, match="Mamba parameter geometry"):
        loader.load_layer(indexes.mamba)


def moe_sources(config, index):
    hidden, count = config.hidden, config.text["n_routed_experts"]
    prefix = f"backbone.layers.{index}.mixer"
    return {f"backbone.layers.{index}.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
            f"{prefix}.gate.weight": torch.full((count, hidden), 1.0, dtype=torch.bfloat16),
            f"{prefix}.gate.e_score_correction_bias": torch.full((count,), 2.0, dtype=torch.bfloat16)}


def test_moe_layer_wires_router_shared_experts_and_expert_loader(lightning_config, tmp_path, indexes):
    config = ModelConfig.from_dict(lightning_config)
    tensors = moe_sources(config, indexes.moe)
    loader = make_loader(lightning_config, tmp_path, tensors)
    loader.load_linear = lambda name: ("linear", name)
    loader.load_fp4_experts = lambda prefixes, **options: ("experts", prefixes, options)
    layer = loader.load_layer(indexes.moe)
    assert layer["router"].shape == (config.text["n_routed_experts"], config.hidden)
    assert layer["correction"].shape == (config.text["n_routed_experts"],)
    assert layer["shared_up"] == ("linear", f"backbone.layers.{indexes.moe}.mixer.shared_experts.up_proj")
    assert layer["shared_down"] == ("linear", f"backbone.layers.{indexes.moe}.mixer.shared_experts.down_proj")
    experts, prefixes, options = layer["experts"]
    assert len(prefixes) == config.text["n_routed_experts"]
    assert options["format"] == "nvfp4" and options["activation"] == "relu2"
    assert options["activation_mode"] == "a16"
    assert options["intermediate_size"] == config.text["moe_intermediate_size"]


def test_moe_layer_rejects_wrong_router_geometry(lightning_config, tmp_path, indexes):
    config = ModelConfig.from_dict(lightning_config)
    tensors = moe_sources(config, indexes.moe)
    tensors[f"backbone.layers.{indexes.moe}.mixer.gate.weight"] = torch.full(
        (config.text["n_routed_experts"] + 1, config.hidden), 1.0, dtype=torch.bfloat16)
    loader = make_loader(lightning_config, tmp_path, tensors)
    with pytest.raises(ValueError, match="router geometry"):
        loader.load_layer(indexes.moe)


def test_rejects_omni_wrapped_checkpoint(lightning_config, tmp_path):
    with pytest.raises(ValueError, match="omni-wrapped"):
        make_loader(lightning_config, tmp_path,
                    {"language_model.backbone.embeddings.weight": torch.empty(1)})


def test_selects_only_text_backbone_and_head_names(lightning_config, tmp_path):
    loader = make_loader(lightning_config, tmp_path, {})
    loader.manifest.tensors = {"backbone.embeddings.weight": None, "lm_head.weight": None,
                               "language_model.backbone.norm_f.weight": None, "mtp.layers.0.A_log": None}
    selected = loader._selected_names()
    assert selected == {"backbone.embeddings.weight", "lm_head.weight"}
