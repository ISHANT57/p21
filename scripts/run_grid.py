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

Plans (grids are defined in eurosat_le.train.grid):
- main: the supervised grid, 3 models x 2 input types x 3 label fractions x 3 seeds = 54 runs.
- tune: the learning-rate sweep (3 learning rates per model on RGB / 10% labels / seed 0). Tuning
  runs never evaluate the test set.
- pl: one pseudo-label student per teacher cell (by default scratch CNN + ResNet-50, 1% and 10%
  labels). A student needs its supervised teacher to be finished; students whose teacher is not
  finished are reported as "waiting" and skipped in this invocation, so run the script again once
  the teachers exist (they may have been trained on another machine).

Behaviour
- The plan is expanded into a list of configs, then --shard k/n keeps entries k, k+n, k+2n, ... of
  that list (0-based). Every shard sees the same ordering, so shards k=0..n-1 together cover each
  run exactly once and do not overlap.
- Runs with metrics.json are counted as done and skipped, so a re-launch after a disconnect resumes
  where the last session stopped. --dry-run only prints the status: [x] done, [w] waiting for the
  teacher, [ ] to do.
- Each run is wrapped in try/except: a failure is printed, appended to results/runs/_failed.log, and
  the grid continues; the exit code is 1 if any run failed. GPU memory is released between runs.
"""

# argparse reads the options; gc, time and traceback are used to free memory, measure elapsed time and log errors.
import argparse
import gc
import time
import traceback

# torch is needed here only to free GPU memory between runs.
import torch

# LABEL_FRACTIONS are the label shares (1%, 10%, 100%); SUBSET_SEEDS are the seeds of the labelled-subset draws.
from eurosat_le.constants import LABEL_FRACTIONS, SUBSET_SEEDS
# parse_set turns "key=value" strings from --set into a dictionary of typed values.
from eurosat_le.train.config import parse_set
# RUNS_DIR is the results/runs folder; run trains one configuration.
from eurosat_le.train.engine import RUNS_DIR, run
# student_config builds the config of a pseudo-label student; train_student trains it from its teacher.
from eurosat_le.ssl.self_training import student_config, train_student
# The grid definitions: default models/inputs/fractions per plan, the tuning learning rates and the grid builders.
from eurosat_le.train.grid import MAIN_MODELS, MODALITIES, PL_FRACTIONS, PL_MODELS, TUNE_LRS, main_grid, tune_grid


def main() -> None:
    """Expand the chosen plan into runs, pick this shard's share, and train every run that is not finished yet.

    Takes no arguments (options come from sys.argv) and returns nothing. With --dry-run only the status is
    printed. Failed runs are logged to results/runs/_failed.log and make the exit code 1.
    """
    # Build the argument parser; the module docstring doubles as the --help text.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Which set of runs to execute: the supervised grid, the learning-rate sweep, or the pseudo-label students.
    p.add_argument("--plan", choices=["main", "tune", "pl"], required=True)
    # Restrict the plan to some models (None means "use the plan's default models").
    p.add_argument("--models", nargs="*", default=None, choices=list(TUNE_LRS),
                   help=f"default: {MAIN_MODELS} (main/tune), {PL_MODELS} (pl)")
    # Restrict the plan to some input types.
    p.add_argument("--modalities", nargs="*", default=MODALITIES, choices=MODALITIES)
    # Restrict the plan to some label fractions (None means "use the plan's default fractions").
    p.add_argument("--fractions", nargs="*", type=float, default=None,
                   help=f"default: {LABEL_FRACTIONS} (main), {PL_FRACTIONS} (pl)")
    # Pseudo-label setting: validation precision the teacher must reach to choose the confidence threshold tau.
    p.add_argument("--target-precision", type=float, default=0.95, help="pl: val precision for choosing tau")
    # Pseudo-label setting: select pseudo-labels class-balanced.
    p.add_argument("--balance", action="store_true", help="pl: class-balanced pseudo-label selection")
    # Which labelled-subset seeds to run.
    p.add_argument("--seeds", nargs="*", type=int, default=SUBSET_SEEDS)
    # Work split "k/n": this process takes every n-th run starting at k, so several machines can share the grid.
    p.add_argument("--shard", default="0/1", help="k/n: run every n-th cell starting at k")
    # Time budget: no new run is started once this many hours have passed.
    p.add_argument("--max-hours", type=float, default=None)
    # Any number of KEY=VALUE overrides applied to every run.
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override fields in every run.")
    # Flag: only list the runs and their status, do not train anything.
    p.add_argument("--dry-run", action="store_true")
    # Parse sys.argv into the args object.
    args = p.parse_args()

    # Convert the --set overrides into a dictionary of typed values.
    overrides = parse_set(args.set)
    # For class-balanced pseudo-labelling, give the students their own name suffix (unless the user set one).
    if args.plan == "pl" and args.balance:
        overrides.setdefault("name_suffix", "plb")  # keep balanced students apart from the default ones
    teachers = {}  # pl plan only: student run name -> teacher run name
    # Expand the chosen plan into a list of run configurations (cfgs).
    if args.plan == "main":
        # Supervised grid: every model x input x label fraction x seed combination.
        cfgs = main_grid(args.models or MAIN_MODELS, args.modalities, args.fractions or LABEL_FRACTIONS,
                         args.seeds, **overrides)
    elif args.plan == "tune":
        # Learning-rate sweep: several learning rates per model on one fixed cell.
        cfgs = tune_grid(args.models or MAIN_MODELS, **overrides)
    else:
        # Pseudo-label plan: start with an empty list and add one student config per planned teacher.
        cfgs = []
        # Students are derived from the planned teacher configs, so the plan can be listed before any teacher exists.
        for t in main_grid(args.models or PL_MODELS, args.modalities, args.fractions or PL_FRACTIONS, args.seeds):
            # Config of the student that learns from teacher t.
            s = student_config(t, **dict(overrides))
            # Remember which teacher belongs to this student (needed later to train it and to check the teacher).
            teachers[s.run_name] = t.run_name
            cfgs.append(s)
    # Split "k/n" into the two integers: this process is shard k out of n.
    k, n = map(int, args.shard.split("/"))
    cfgs = cfgs[k::n]  # shard k of n: every n-th run starting at index k

    def finished(name):
        """Return True if the run called `name` has completed (its metrics.json exists)."""
        # metrics.json is the last file a run writes, so it marks a completed run
        return (RUNS_DIR / name / "metrics.json").exists()

    # Sort this shard's runs into: already done, still to do, and students waiting for an unfinished teacher.
    done = [c for c in cfgs if finished(c.run_name)]
    todo = [c for c in cfgs if c not in done]
    waiting = [c for c in todo if c.run_name in teachers and not finished(teachers[c.run_name])]
    # Waiting students cannot be trained yet, so take them out of the to-do list.
    todo = [c for c in todo if c not in waiting]
    # Summary line: how many runs in this shard, how many done, how many to do (and waiting, for the pl plan).
    print(f"{args.plan} plan, shard {k}/{n}: {len(cfgs)} runs, {len(done)} done, {len(todo)} to do"
          + (f", {len(waiting)} waiting for their teacher run" if teachers else ""))
    # Dry run: list every run with a status mark ([x] done, [w] waiting, [ ] to do) and stop without training.
    if args.dry_run:
        for c in cfgs:
            mark = "x" if c in done else "w" if c in waiting else " "
            print(f"  [{mark}] {c.run_name}" + (f"  <- {teachers[c.run_name]}" if c.run_name in teachers else ""))
        return

    # Start the clock for the time budget and prepare the list of failed run names.
    t0, failed = time.time(), []
    # Work through the unfinished runs one after another, counting from 1 for the progress display.
    for i, cfg in enumerate(todo, 1):
        # Hours elapsed since the grid started.
        hours = (time.time() - t0) / 3600
        # Checked only between runs: a run that has started is never interrupted.
        if args.max_hours is not None and hours >= args.max_hours:
            print(f"time budget of {args.max_hours} h used; stopping before {cfg.run_name}")
            break
        # Progress header: run number, run name and elapsed time.
        print(f"\n=== [{i}/{len(todo)}] {cfg.run_name}  (elapsed {hours:.2f} h) ===", flush=True)
        try:
            # A pseudo-label student is trained from its teacher; any other run is trained directly.
            if cfg.run_name in teachers:
                train_student(teachers[cfg.run_name], args.target_precision, args.balance, **overrides)
            else:
                run(cfg)
        except Exception:  # keep going; one broken cell should not waste a GPU session
            # Remember the failure and print the full error to the screen.
            failed.append(cfg.run_name)
            traceback.print_exc()
            # Also append the time, run name and error to a log file so it can be read after the session.
            with open(RUNS_DIR / "_failed.log", "a") as f:
                f.write(f"{time.strftime('%F %T')} {cfg.run_name}\n{traceback.format_exc()}\n")
        finally:
            # Whatever happened, release Python objects and cached GPU memory before the next run.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # Final summary: total time and the names of failed runs.
    print(f"\nfinished in {(time.time() - t0) / 3600:.2f} h; failed: {failed or 'none'}")
    # Non-zero exit status if any run failed, so calling scripts notice.
    if failed:
        raise SystemExit(1)


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()
