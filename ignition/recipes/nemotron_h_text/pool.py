"""Mamba continuations and paged FP8 key/value histories, shared with the omni recipe.

The checkpoint declares ``mamba_ssm_cache_dtype: float32`` and ships
calibrated FP8 key/value scales. This family (both Nemotron-H recipes)
instead stores BF16 recurrent/convolution state and applies unit scales —
the same contract as ``nemotron_h`` — so the precision deviation is a
family-wide decision, not a per-recipe one; the serving-parity run in
``qualify/serving_parity.py`` is the qualification evidence for it.
"""

from ..nemotron_h.pool import (REPLAY_RING, ContinuationPool, continuation_layout, page_bytes,
                               page_specs)

__all__ = ["REPLAY_RING", "ContinuationPool", "continuation_layout", "page_bytes", "page_specs"]
