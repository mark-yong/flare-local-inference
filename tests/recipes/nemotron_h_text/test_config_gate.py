"""Recipe gate: which NemotronH checkpoints nemotron_h_text accepts, and derived geometry."""

from types import SimpleNamespace

import pytest

from ignition.recipes.nemotron_h_text.config import ModelConfig


def config(lightning_config, **changes):
    return ModelConfig.from_dict({**lightning_config, **changes})


def test_accepts_lightning_checkpoint(lightning_config):
    model = config(lightning_config)
    counts = {kind: model.layers.count(kind) for kind in set(model.layers)}
    assert counts == {"mamba": 23, "moe": 23, "attention": 6}
    assert len(model.layers) == model.text["num_hidden_layers"]
    assert model.text["routed_scaling_factor"] == 2.5
    assert model.hidden == model.text["hidden_size"]
    assert model.vocabulary == model.text["vocab_size"]
    assert model.eos_token == lightning_config["eos_token_id"]
    assert model.mamba_width == model.text["mamba_num_heads"] * model.text["mamba_head_dim"]
    assert model.convolution_width == model.mamba_width + 2 * model.text["n_groups"] * model.text["ssm_state_size"]


def test_rejects_omni_model_type(lightning_config):
    with pytest.raises(ValueError, match="NemotronHForCausalLM"):
        config(lightning_config, model_type="nemotron_h_puzzle")


def test_rejects_foreign_architecture(lightning_config):
    with pytest.raises(ValueError, match="NemotronHForCausalLM"):
        config(lightning_config, architectures=["NemotronHOtherForCausalLM"])


def test_rejects_block_configs(lightning_config):
    with pytest.raises(ValueError, match="block configurations"):
        config(lightning_config, block_configs=[{"moe": {}}])


def test_rejects_vision_encoders(lightning_config):
    with pytest.raises(ValueError, match="vision"):
        config(lightning_config, vision_config={})


def test_rejects_shared_expert_count(lightning_config):
    with pytest.raises(ValueError, match="n_shared_experts"):
        config(lightning_config, n_shared_experts=2)


def test_rejects_unsupported_mamba_geometry(lightning_config):
    with pytest.raises(ValueError, match="geometry"):
        config(lightning_config, mamba_head_dim=96)


def test_rejects_incomplete_nvfp4_blocks(lightning_config):
    with pytest.raises(ValueError, match="NVFP4"):
        config(lightning_config, moe_intermediate_size=1857)


@pytest.mark.parametrize("scale", [0, -2.5, "2.5", None])
def test_rejects_bad_routed_scaling_factor(lightning_config, scale):
    with pytest.raises(ValueError, match="routed_scaling_factor"):
        config(lightning_config, routed_scaling_factor=scale)


def test_rejects_layer_sequence_mismatch(lightning_config):
    with pytest.raises(ValueError, match="layer sequence"):
        config(lightning_config, layers_block_type=lightning_config["layers_block_type"][:-1])


def test_rejects_unknown_block_kind(lightning_config):
    with pytest.raises(ValueError, match="block"):
        kinds = lightning_config["layers_block_type"]
        config(lightning_config, layers_block_type=["conv"] + kinds[1:])


def test_rejects_non_positive_field(lightning_config):
    with pytest.raises(ValueError, match="positive"):
        config(lightning_config, hidden_size=0)


def test_rejects_multi_gpu_shard(lightning_config):
    model = config(lightning_config)
    with pytest.raises(ValueError, match="one GPU"):
        model.validate_shard(SimpleNamespace(world_size=2))
