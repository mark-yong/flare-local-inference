"""Nemotron-H text heterogeneous residual blocks and vocabulary projection."""

from dataclasses import dataclass

import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.paged_attention import PagedAttentionMetadata
from ignition.components.recurrent import DeltaMetadata
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import PreparedAttention
from .config import PAGE_TOKENS
from .mamba import PreparedMamba
from .moe import PreparedRoutedMoE


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    normalized: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, buffers):
        def zeros(shape):
            return buffers.zeros(shape, torch.bfloat16)
        return cls(zeros((tokens, config.hidden)), zeros((tokens, config.hidden)),
            zeros((rows, config.hidden)), zeros((rows, config.vocabulary)), zeros((rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.normalized[:tokens], self.hidden[:rows],
                          self.local_logits[:rows], self.logits[:rows])


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, session, name, workspace,
                 layer_buffers, coordinator=None, vocabulary_runtime=None, buffers=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.cache_metadata = PagedAttentionMetadata(metadata, page_tokens=PAGE_TOKENS, buffers=buffers)
        config, source = weights.config, scratch.normalized
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.activations,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.delta = DeltaMetadata.allocate(metadata.starts.numel(), source.device)
        self.kv_scale = torch.ones(1, device=source.device, dtype=torch.float32)
        self.layers = []
        previous = None
        for index, w in enumerate(weights.layers):
            prefix, kind = f"{name}.layer{index}", config.layers[index]
            buffers = layer_buffers.cursor()
            norm = PreparedRMSNorm(scratch.activations, w["norm"], source,
                eps=config.text["layer_norm_epsilon"], residual=previous, session=session, name=prefix + ".norm")
            kwargs = dict(session=session, name=prefix, workspace=workspace, buffers=buffers)
            if kind == "mamba":
                block = PreparedMamba(w, source, pool.recurrent(index), self.delta, config, mode=mode, **kwargs)
            elif kind == "attention":
                key, value = pool.attention(index, config)
                block = PreparedAttention(w, source, metadata, key, value, config, mode=mode,
                    key_scale=self.kv_scale, value_scale=self.kv_scale, cache_metadata=self.cache_metadata, **kwargs)
            else:
                block = PreparedRoutedMoE(w, source, config, index,
                    num_valid_tokens=metadata.live_counts[:1], **kwargs)
            # Every mixer returns a contiguous hidden-width matrix consumed by the following norm.
            previous = source
            self.layers.append((norm, block))
        self.final = PreparedRMSNorm(scratch.activations, weights.norm, source,
            eps=config.text["layer_norm_epsilon"], residual=previous, session=session, name=name + ".norm_f")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto")

    def __call__(self):
        m, d = self.metadata, self.delta
        self.cache_metadata()
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        d.fresh.copy_(m.starts == 0)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        residual, output = self.embedding(), None
        for index, (norm, block) in enumerate(self.layers):
            normalized = norm(residual, output)
            output = block(normalized)
        normalized = self.final(residual, output)
        self.scratch.hidden.copy_(normalized[m.sample_indices])
        self.head(self.scratch.hidden)
