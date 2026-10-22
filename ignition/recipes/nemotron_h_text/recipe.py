"""Nemotron 3.5 Lightning 30B-A3B text preparation capacities and serving defaults."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None or plan.speculation is not None:
        raise ValueError("Nemotron-H text preparation supports one local GPU and ordinary text generation")
    if plan.prompt_logprobs:
        raise ValueError("Nemotron-H text does not support prompt logprobs")
    config = PreparationConfig(compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort,
        tuning_replica=plan.tuning_replica,
        max_active=plan.concurrency, context=plan.context, stream_overlap=plan.stream_overlap,
        prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache)
    return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


RECIPE = Recipe(
    name="nemotron_h_text", served_model_name="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    prepare=prepare, usage=usage,
    # Text sampling: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1, min_p=0.,
        repetition_penalty=1., presence_penalty=0., frequency_penalty=0.),
    checkpoints=(Checkpoint(alias="nemotron-lightning-30b",
        repository="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        revision="1de6d84feaf74a2c8c4a6f4f3150f4dc2204cccd", architecture="NemotronHForCausalLM",
        # stored_bytes is the full checkpoint index total; device_bytes is the measured
        # device-resident "Weights / numeric constants" from the completed export's memory
        # report (18,040.76 MiB, MiB-rounded, rounded up to the kilobyte). The in-checkpoint
        # MTP module is not loaded, so device bytes sit below stored bytes.
        stored_bytes=21_559_589_596, device_bytes=18_917_108_000),),
    context=262144, page_tokens=128, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    max_draft_tokens=0,
    # Single-rank costs: 23 Mamba layers of (64x64x128 recurrent + 6144x3 convolution) BF16
    # rows plus six 2x128-head FP8 KV pages, both derived from the checkpoint geometry and
    # pinned by tests/recipes/nemotron_h_text/test_layout.py. The 4 GiB graph budget matches
    # the nemotron_h sibling and covers the completed export's measured pools (graph scratch
    # 785 MiB, graph capture 272 MiB, operator scratch 307 MiB).
    ranks=MappingProxyType({1: RankCosts(25_363_968, 393_216, 4 << 30)}),
)
