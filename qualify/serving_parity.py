"""Cross-engine serving parity for a prepared package against a reference engine.

Weight-wiring evidence for a recipe candidate: the same checkpoint served by
two independent implementations. Exact greedy parity through the API is not
available — the server rejects temperature=0 (greedy-decoding guard) — so the
comparison uses near-greedy sampling with a fixed seed. Even then, different
kernels disagree on low-margin tokens: the agreement rate is evidence to
judge, not a numerical-tolerance gate. A mis-wired MoE branch or a wrong
routing scale collapses agreement; correct weights keep it high under
near-greedy decoding.

The baseline is either a live engine serving the same checkpoint
(--baseline-url) or a recorded run (--baseline-json, shape
{"temperature": ..., "outputs": {prompt: completion_text, ...}}).

Examples:
    python -m qualify.serving_parity --candidate http://HOST:PORT \
        --baseline-json baseline.json --min-agree 0.9 --out parity.json
"""

import argparse
import json
from pathlib import Path

import requests

DEFAULT_PROMPTS = Path(__file__).parent / "serving_parity_prompts.txt"


def complete(base, model, prompt, *, temperature, top_p, seed, max_tokens):
    response = requests.post(f"{base}/v1/completions", json={
        "model": model, "prompt": prompt, "temperature": temperature, "top_p": top_p,
        "seed": seed, "max_tokens": max_tokens}, timeout=300)
    response.raise_for_status()
    return response.json()["choices"][0]["text"]


def model_id(base):
    return requests.get(f"{base}/v1/models", timeout=30).json()["data"][0]["id"]


def divergence_at(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Cross-engine serving parity for a prepared package against a reference engine.")
    parser.add_argument("--candidate", required=True, help="URL of the flare server under test")
    parser.add_argument("--baseline-url", help="live reference engine serving the same checkpoint")
    parser.add_argument("--baseline-json", help="recorded baseline run (JSON)")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, help="write full per-prompt results here")
    parser.add_argument("--min-agree", type=float, help="fail below this agreement rate (default: report only)")
    args = parser.parse_args()

    if not args.baseline_url and not args.baseline_json:
        parser.error("one of --baseline-url or --baseline-json is required")
    prompts = [line.strip() for line in args.prompts.read_text().splitlines()
               if line.strip() and not line.strip().startswith("#")]

    candidate_model = model_id(args.candidate)
    if args.baseline_json:
        recorded = json.loads(Path(args.baseline_json).read_text())
        if abs(recorded.get("temperature", args.temperature) - args.temperature) > 1e-6:
            raise SystemExit("recorded baseline temperature does not match --temperature")
        baseline_outputs = recorded["outputs"]
    else:
        baseline_model = model_id(args.baseline_url)
        baseline_outputs = {p: complete(args.baseline_url, baseline_model, p,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            max_tokens=args.max_tokens) for p in prompts}

    results, identical = [], 0
    for prompt in prompts:
        candidate = complete(args.candidate, candidate_model, prompt,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            max_tokens=args.max_tokens)
        baseline = baseline_outputs[prompt]
        match = candidate.strip() == baseline.strip()
        identical += match
        results.append({"prompt": prompt, "candidate": candidate, "baseline": baseline,
                        "identical": match, "divergence_at": divergence_at(candidate, baseline)})

    agreement = identical / len(prompts)
    divergences = [r["divergence_at"] for r in results if not r["identical"]]
    print(f"parity: {identical}/{len(prompts)} prompts token-identical "
          f"(temperature {args.temperature}, seed {args.seed})")
    for r in results[:5]:
        print(f"  {r['prompt'][:48]!r}: identical={r['identical']} divergence_at={r['divergence_at']}")
    if not identical and divergences:
        print(f"  mean first-divergence over mismatches: "
              f"{sum(divergences) / len(divergences):.0f} chars")
    if args.out:
        args.out.write_text(json.dumps({
            "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
            "max_tokens": args.max_tokens, "agreement": agreement, "results": results}, indent=2))
    if args.min_agree is not None and agreement < args.min_agree:
        raise SystemExit(f"agreement {agreement:.2f} < required {args.min_agree}")


if __name__ == "__main__":
    main()
