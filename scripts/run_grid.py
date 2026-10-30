"""Run many training cells in sequence; finished cells are skipped, so it resumes after a disconnect.

Usage:
    python scripts/run_grid.py --plan tune --dry-run                 # list the LR sweep and its status
    python scripts/run_grid.py --plan tune --models resnet50          # run the sweep for one model
    python scripts/run_grid.py --plan main --dry-run                  # 54-run supervised grid status
    python scripts/run_grid.py --plan main --models resnet50 --modalities rgb --max-hours 11
    python scripts/run_grid.py --plan main --shard 0/2                # split the grid across two machines

--shard k/n takes every n-th run starting at k, so group members (or a Colab and a
Kaggle session) can work on disjoint parts of the grid at the same time.
--max-hours stops *starting* new runs once the budget is used (Kaggle sessions end at 12 h).
"""

import argparse
import gc
import time
import traceback

import torch

from eurosat_le.constants import LABEL_FRACTIONS, SUBSET_SEEDS
from eurosat_le.train.config import parse_set
from eurosat_le.train.engine import RUNS_DIR, run
from eurosat_le.train.grid import MAIN_MODELS, MODALITIES, TUNE_LRS, main_grid, tune_grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", choices=["main", "tune"], required=True)
    p.add_argument("--models", nargs="*", default=MAIN_MODELS, choices=list(TUNE_LRS))
    p.add_argument("--modalities", nargs="*", default=MODALITIES, choices=MODALITIES)
    p.add_argument("--fractions", nargs="*", type=float, default=LABEL_FRACTIONS)
    p.add_argument("--seeds", nargs="*", type=int, default=SUBSET_SEEDS)
    p.add_argument("--shard", default="0/1", help="k/n: run every n-th cell starting at k")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override fields in every run.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    overrides = parse_set(args.set)
    if args.plan == "main":
        cfgs = main_grid(args.models, args.modalities, args.fractions, args.seeds, **overrides)
    else:
        cfgs = tune_grid(args.models, **overrides)
    k, n = map(int, args.shard.split("/"))
    cfgs = cfgs[k::n]

    done = [c for c in cfgs if (RUNS_DIR / c.run_name / "metrics.json").exists()]
    todo = [c for c in cfgs if c not in done]
    print(f"{args.plan} plan, shard {k}/{n}: {len(cfgs)} runs, {len(done)} done, {len(todo)} to do")
    if args.dry_run:
        for c in cfgs:
            print(f"  [{'x' if c in done else ' '}] {c.run_name}")
        return

    t0, failed = time.time(), []
    for i, cfg in enumerate(todo, 1):
        hours = (time.time() - t0) / 3600
        if args.max_hours is not None and hours >= args.max_hours:
            print(f"time budget of {args.max_hours} h used; stopping before {cfg.run_name}")
            break
        print(f"\n=== [{i}/{len(todo)}] {cfg.run_name}  (elapsed {hours:.2f} h) ===", flush=True)
        try:
            run(cfg)
        except Exception:  # keep going; one broken cell should not waste a GPU session
            failed.append(cfg.run_name)
            traceback.print_exc()
            with open(RUNS_DIR / "_failed.log", "a") as f:
                f.write(f"{time.strftime('%F %T')} {cfg.run_name}\n{traceback.format_exc()}\n")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\nfinished in {(time.time() - t0) / 3600:.2f} h; failed: {failed or 'none'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
