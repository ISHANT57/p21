"""Run many training cells in sequence; finished cells are skipped, so it resumes after a disconnect.

Usage:
    python scripts/run_grid.py --plan tune --dry-run                 # list the LR sweep and its status
    python scripts/run_grid.py --plan tune --models resnet50          # run the sweep for one model
    python scripts/run_grid.py --plan main --dry-run                  # 54-run supervised grid status
    python scripts/run_grid.py --plan main --models resnet50 --modalities rgb --max-hours 11
    python scripts/run_grid.py --plan main --shard 0/2                # split the grid across two machines
    python scripts/run_grid.py --plan pl --dry-run                    # pseudo-label students (needs teachers)
    python scripts/run_grid.py --plan pl --balance --models resnet50 --fractions 0.01   # class-balanced (_plb)

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
from eurosat_le.ssl.self_training import student_config, train_student
from eurosat_le.train.grid import MAIN_MODELS, MODALITIES, PL_FRACTIONS, PL_MODELS, TUNE_LRS, main_grid, tune_grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", choices=["main", "tune", "pl"], required=True)
    p.add_argument("--models", nargs="*", default=None, choices=list(TUNE_LRS),
                   help=f"default: {MAIN_MODELS} (main/tune), {PL_MODELS} (pl)")
    p.add_argument("--modalities", nargs="*", default=MODALITIES, choices=MODALITIES)
    p.add_argument("--fractions", nargs="*", type=float, default=None,
                   help=f"default: {LABEL_FRACTIONS} (main), {PL_FRACTIONS} (pl)")
    p.add_argument("--target-precision", type=float, default=0.95, help="pl: val precision for choosing tau")
    p.add_argument("--balance", action="store_true", help="pl: class-balanced pseudo-label selection")
    p.add_argument("--seeds", nargs="*", type=int, default=SUBSET_SEEDS)
    p.add_argument("--shard", default="0/1", help="k/n: run every n-th cell starting at k")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override fields in every run.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    overrides = parse_set(args.set)
    if args.plan == "pl" and args.balance:
        overrides.setdefault("name_suffix", "plb")  # keep balanced students apart from the default ones
    teachers = {}
    if args.plan == "main":
        cfgs = main_grid(args.models or MAIN_MODELS, args.modalities, args.fractions or LABEL_FRACTIONS,
                         args.seeds, **overrides)
    elif args.plan == "tune":
        cfgs = tune_grid(args.models or MAIN_MODELS, **overrides)
    else:
        cfgs = []
        for t in main_grid(args.models or PL_MODELS, args.modalities, args.fractions or PL_FRACTIONS, args.seeds):
            s = student_config(t, **dict(overrides))
            teachers[s.run_name] = t.run_name
            cfgs.append(s)
    k, n = map(int, args.shard.split("/"))
    cfgs = cfgs[k::n]

    def finished(name):
        return (RUNS_DIR / name / "metrics.json").exists()

    done = [c for c in cfgs if finished(c.run_name)]
    todo = [c for c in cfgs if c not in done]
    waiting = [c for c in todo if c.run_name in teachers and not finished(teachers[c.run_name])]
    todo = [c for c in todo if c not in waiting]
    print(f"{args.plan} plan, shard {k}/{n}: {len(cfgs)} runs, {len(done)} done, {len(todo)} to do"
          + (f", {len(waiting)} waiting for their teacher run" if teachers else ""))
    if args.dry_run:
        for c in cfgs:
            mark = "x" if c in done else "w" if c in waiting else " "
            print(f"  [{mark}] {c.run_name}" + (f"  <- {teachers[c.run_name]}" if c.run_name in teachers else ""))
        return

    t0, failed = time.time(), []
    for i, cfg in enumerate(todo, 1):
        hours = (time.time() - t0) / 3600
        if args.max_hours is not None and hours >= args.max_hours:
            print(f"time budget of {args.max_hours} h used; stopping before {cfg.run_name}")
            break
        print(f"\n=== [{i}/{len(todo)}] {cfg.run_name}  (elapsed {hours:.2f} h) ===", flush=True)
        try:
            if cfg.run_name in teachers:
                train_student(teachers[cfg.run_name], args.target_precision, args.balance, **overrides)
            else:
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
