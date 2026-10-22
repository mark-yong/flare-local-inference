"""State layout arithmetic: the recipe's declared rank costs must match derived geometry.

``state_row_bytes`` and ``page_bytes`` are derived from the checkpoint
config's Mamba and KV geometry; this test fails when the declared values
drift from the derivation. The 4 GiB graph budget matches the ``nemotron_h``
sibling and covers the completed export's measured pools (graph scratch
785 MiB, graph capture 272 MiB, operator scratch 307 MiB).
"""

from ignition.recipes.nemotron_h_text.config import ModelConfig
from ignition.recipes.nemotron_h_text.pool import continuation_layout, page_bytes
from ignition.recipes.nemotron_h_text.recipe import RECIPE


def test_declared_costs_match_derived_layout(lightning_config):
    config = ModelConfig.from_dict(lightning_config)
    declared = RECIPE.ranks[1]
    assert continuation_layout(config).row_bytes == 25_363_968 == declared.state_row_bytes
    assert page_bytes(config) == 393_216 == declared.page_bytes


def test_single_rank_recipe(lightning_config):
    assert tuple(RECIPE.ranks) == (1,)
