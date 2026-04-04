#!/home/setupishe/bel_conf/.venv/bin/python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def load_last_row(run_dir: Path) -> dict[str, float | str]:
    csv_path = run_dir / "results.csv"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    row = rows[-1]
    parsed: dict[str, float | str] = {}
    for key, value in row.items():
        value = value.strip()
        try:
            parsed[key.strip()] = float(value)
        except ValueError:
            parsed[key.strip()] = value
    return parsed


def load_state_dict(run_dir: Path) -> dict[str, torch.Tensor]:
    ckpt_path = run_dir / "weights" / "last.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_obj = ckpt.get("ema") or ckpt.get("model")
    if model_obj is None:
        raise RuntimeError(f"No model weights found in {ckpt_path}")
    return model_obj.state_dict()


def compare_state_dicts(
    baseline_sd: dict[str, torch.Tensor],
    candidate_sd: dict[str, torch.Tensor],
) -> tuple[bool, int, float, str | None]:
    base_keys = set(baseline_sd)
    cand_keys = set(candidate_sd)
    if base_keys != cand_keys:
        missing = sorted(base_keys - cand_keys)[:5]
        extra = sorted(cand_keys - base_keys)[:5]
        return False, -1, float("inf"), f"key mismatch missing={missing} extra={extra}"

    differing_tensors = 0
    max_abs_diff = 0.0
    first_diff = None

    for key in sorted(base_keys):
        base_tensor = baseline_sd[key]
        cand_tensor = candidate_sd[key]
        if base_tensor.shape != cand_tensor.shape:
            return False, -1, float("inf"), f"shape mismatch for {key}: {tuple(base_tensor.shape)} vs {tuple(cand_tensor.shape)}"
        if not torch.equal(base_tensor, cand_tensor):
            differing_tensors += 1
            diff = (base_tensor.float() - cand_tensor.float()).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            if first_diff is None:
                first_diff = key

    return differing_tensors == 0, differing_tensors, max_abs_diff, first_diff


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two Ultralytics training runs for strict parity.")
    parser.add_argument("baseline_run_dir", type=Path)
    parser.add_argument("candidate_run_dir", type=Path)
    args = parser.parse_args()

    baseline_run = args.baseline_run_dir.resolve()
    candidate_run = args.candidate_run_dir.resolve()

    baseline_row = load_last_row(baseline_run)
    candidate_row = load_last_row(candidate_run)

    print(f"baseline:  {baseline_run}")
    print(f"candidate: {candidate_run}")
    print()
    print("last results.csv row deltas:")
    all_keys = sorted(set(baseline_row) | set(candidate_row))
    for key in all_keys:
        base_value = baseline_row.get(key)
        cand_value = candidate_row.get(key)
        if isinstance(base_value, float) and isinstance(cand_value, float):
            delta = cand_value - base_value
            print(f"  {key}: baseline={base_value:.6f} candidate={cand_value:.6f} delta={delta:+.6f}")
        else:
            print(f"  {key}: baseline={base_value} candidate={cand_value}")

    identical, differing_tensors, max_abs_diff, detail = compare_state_dicts(
        load_state_dict(baseline_run),
        load_state_dict(candidate_run),
    )

    print()
    print("EMA checkpoint comparison:")
    print(f"  identical_weights: {identical}")
    print(f"  differing_tensors: {differing_tensors}")
    print(f"  max_abs_diff: {max_abs_diff}")
    print(f"  first_difference: {detail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
