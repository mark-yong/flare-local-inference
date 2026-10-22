"""Heterogeneous routed ReLU-squared experts with a full-width shared expert."""

import torch

from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import PreparedUngatedMLP
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.preparation.scratch import ScratchArena


class PreparedRoutedMoE:
    def __init__(self, weights, source, config, index, *, session, name, workspace, buffers,
                 num_valid_tokens=None):
        c, tokens = config.text, source.shape[0]

        def zeros(width, dtype=torch.bfloat16):
            return buffers.zeros((tokens, width), dtype)

        logits = zeros(c["n_routed_experts"], torch.float32)
        routes = RouterScratch.allocate(tokens, source.device, experts=c["n_routed_experts"],
            top_k=c["num_experts_per_tok"], buffers=buffers)
        self.router_projection = UnquantizedProjection(source, weights["router"], logits,
            session=session, name=name + ".router_logits")
        self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
            renormalize=True, routed_scaling_factor=c["routed_scaling_factor"],
            num_valid_tokens=num_valid_tokens, session=session, name=name + ".route")
        output = zeros(config.hidden)
        self.experts = PreparedRoutedExperts(source, weights["experts"], routes.ids, routes.weights, output,
            session=session, name=name + ".experts", workspace=workspace)
        shared_width = c["moe_shared_expert_intermediate_size"]
        self.shared = PreparedUngatedMLP(source, weights["shared_up"], weights["shared_down"],
            zeros(shared_width), zeros(shared_width), zeros(config.hidden), activation="relu2",
            session=session, name=name + ".shared", workspace=workspace)
        self.concurrent_shared = None
        if tokens <= 64 and isinstance(workspace, ScratchArena) and workspace.overlap:
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.router_projection.plan, self.router.plan, self.experts.plan),
                session=session, workspace=workspace, defer_launch=1 < tokens <= 4)

    def __call__(self, source):
        if self.concurrent_shared is not None:
            self.concurrent_shared.start(source)
        self.router(self.router_projection(source))
        routed = self.experts()
        shared = (self.shared(source) if self.concurrent_shared is None
                  else self.concurrent_shared.finish(source, routed))
        return routed + shared
