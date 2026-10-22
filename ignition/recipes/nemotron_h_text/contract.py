"""ChatML frontend and shared continuation and paged key/value storage contracts."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p

from .config import PAGE_TOKENS

FRONTEND = c.frontend(tools=p.QWEN_CODER, terminal_markers=["<tool_call>"],
    assistant_marker="<|im_start|>assistant\n", message_end_markers=["<|im_end|>"],
    prefix_boundary=c.prefix_boundary(["system", "developer"], append_role="user", minimum_messages=0,
        required_start="<|im_start|>system\n", required_marker="<|im_end|>"))
FRONTEND["tools"]["wrapping_newlines"] = 1


def prepare(signatures, buffers, pool, *, vocabulary):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()}}
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    regions = [c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_KEY_VALUE)
               for name, tensor in pool.pages.items()]
    return c.model_contract(vocabulary=vocabulary, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation,
                                 pending=c.state_view("request_pending_token", 8)),
        frontend=FRONTEND)
