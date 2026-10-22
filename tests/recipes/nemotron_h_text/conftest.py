"""Shared fixtures for the nemotron_h_text recipe tests."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def lightning_config():
    """Checkpoint config.json fields the recipe gate, loader, and layout tests read.

    Values are verbatim from config.json at revision
    1de6d84feaf74a2c8c4a6f4f3150f4dc2204cccd. The full file also carries a
    ~30,000-line ``quantization_config`` that no Python code reads (export
    reads weight formats from the safetensors manifest), and the rotary
    fields the reference forward never uses; both are omitted here.
    """
    return json.loads((FIXTURES / "lightning-30b-config.json").read_text())
