"""Mamba-2 projection, biased causal convolution, SSD, and grouped gated norm.

Forked from the Bolt-family mamba block: Nemotron 3.5 text checkpoints keep
their mamba input and output projections in tensor-scaled FP8, which the
dedicated tensor-FP8 projection executes; Bolt's block-scaled formats keep
using the shared quantized projection.
"""

import torch
from b12x.sequence import short_conv

from ignition.components.gated_norm import PreparedGatedRMSNorm
from ignition.components.linear import prepare_projection
from ignition.components.mamba import PreparedMambaScan, PreparedMambaUpdate
from ignition.components.recurrent import prepare_sequence_binding


class PreparedMamba:
    def __init__(self, weights, source, state, metadata, config, *, mode, session, name, workspace, buffers):
        c, tokens = config.text, source.shape[0]
        heads, dim, n, groups = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
        width, channels = config.mamba_width, config.convolution_width

        def buffer(columns):
            return buffers.zeros((tokens, columns), torch.bfloat16)

        projected = buffer(width + channels + heads)
        gate, xbc, dt = projected.split((width, channels, heads), dim=-1)
        mixed, output = buffer(channels), buffer(config.hidden)
        recurrence, gated = buffer(width), buffer(width)
        self.input = prepare_projection(source, weights["input"], projected, session=session,
            name=name + ".input", workspace=workspace)
        self.output = prepare_projection(gated, weights["output"], output, session=session,
            name=name + ".output", workspace=workspace)
        sequences = metadata.query_start_loc.numel() - 1
        args = dict(x=xbc, conv_weight=weights["convolution"], conv_bias=weights["convolution_bias"],
            query_start_loc=metadata.query_start_loc, state_slot_ids=metadata.convolution_slots,
            state_is_fresh=metadata.fresh, num_accepted_tokens=metadata.accepted_tokens,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            conv_state=state.convolution, out=mixed)
        caps = short_conv.Caps(device=source.device, mode=mode, channels=channels, kernel_size=c["conv_kernel"],
            max_tokens=tokens, max_seqs=sequences, max_state_slots=state.convolution.shape[0],
            max_speculative_tokens=state.convolution.shape[-1] - c["conv_kernel"] + 1, round_before_silu=False)
        plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**args))
        prepare_sequence_binding(plan, args, session=session, name=name + ".convolution", mode=mode,
            public_bind=short_conv.bind, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: setattr(self, "convolution", binding))
        x, B, C = mixed.split((width, groups * n, groups * n), dim=-1)
        args = dict(x=x.view(tokens, heads, dim), B=B.view(tokens, groups, n), C=C.view(tokens, groups, n),
            dt=dt, A=weights["A"], D=weights["D"], dt_bias=weights["dt_bias"], recurrent_state=state.recurrent,
            cu_seqlens=metadata.query_start_loc, initial_state_indices=metadata.initial_state_indices,
            final_state_indices=metadata.final_state_indices, state_is_fresh=metadata.fresh,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens, out=recurrence.view(tokens, heads, dim))
        if mode == "decode":
            self.recurrence = PreparedMambaUpdate(args, session=session, name=name + ".state_update")
        else:
            self.recurrence = PreparedMambaScan(args, session=session, name=name + ".scan", workspace=workspace)
        self.norm = PreparedGatedRMSNorm(recurrence.view(tokens, groups, width // groups),
            gate.view(tokens, groups, width // groups), weights["output_norm"].view(groups, width // groups),
            gated.view(tokens, groups, width // groups), metadata.num_tokens, eps=c["layer_norm_epsilon"],
            activation="silu", norm_before_gate=False, session=session, name=name + ".gated_norm")
        self.gated = gated

    def __call__(self, source):
        self.input(source)
        short_conv.run(self.convolution)
        self.recurrence()
        self.norm()
        return self.output(self.gated)
