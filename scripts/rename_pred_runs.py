#!/usr/bin/env python3
"""Rename existing experiments/<BENCHMARK>/pred/ run folders to the current scheme.

The naming was changed to mirror the config files:

    causal_lookahead_<tag>_<B>_chunk_<cs>_M_..._lH_<x>_eta_<e>_..._tau_<t>
      ->  lorekv-<cell>-<brule>-<sel>[-<model>]_B_<B>_chunk_<cs>_M_..._kappa_<k>

The <cell> is computed from the run's ACTUAL coefficients, translating the old
(lambda_H, eta) parametrization through Proposition 1:

    (beta, gamma, lambda_D) = (eta, eta * (1 + lambda_H), eta * lambda_D)

so an old `lH_0.0` run is correctly relabelled `logp` (gamma = 1), not "no penalty".
Baseline folders (andpro / critical / causal_forward_only) are left alone.

Symlinked folders -- runs already archived to scratch -- are renamed in place; only
the link is touched, never its target.

Usage (from the repo root):
    python scripts/rename_pred_runs.py                 # dry run: print the plan
    python scripts/rename_pred_runs.py --apply         # do it
    python scripts/rename_pred_runs.py --exp RULER --apply
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_naming import canonical_name, parse_run_name  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append",
                    help="benchmark to process (repeatable); default: all three")
    ap.add_argument("--pred-dir-name", default="pred",
                    help="subfolder under experiments/<exp> (also try pred_e)")
    ap.add_argument("--apply", action="store_true", help="actually rename")
    args = ap.parse_args()
    exps = args.exp or ["LongBench", "RULER", "NIAH"]

    total = renamed = skipped = 0
    for exp in exps:
        root = os.path.join(REPO_ROOT, "experiments", exp, args.pred_dir_name)
        if not os.path.isdir(root):
            print(f"[{exp}] no {args.pred_dir_name}/ -- skipping")
            continue

        plan, unchanged, baselines, unparsed = [], 0, 0, []
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not (os.path.isdir(path) or os.path.islink(path)):
                continue
            total += 1
            p = parse_run_name(name)
            if p is None:
                unparsed.append(name)
                continue
            if p.get("mode") != "causal_lookahead":
                baselines += 1
                continue
            new = canonical_name(name)
            if new is None:
                unparsed.append(name)
            elif new == name:
                unchanged += 1
            else:
                plan.append((name, new))

        # A rename must never clobber an existing folder. Collisions cannot happen by
        # construction (the new name is injective in the parameters), but check anyway:
        # two DIFFERENT old folders mapping to one new name would mean silent data loss.
        seen = {}
        collisions = []
        for old, new in plan:
            seen.setdefault(new, []).append(old)
        for new, olds in seen.items():
            if len(olds) > 1:
                collisions.append((new, olds))
            elif os.path.exists(os.path.join(root, new)) and new not in {o for o, _ in plan}:
                collisions.append((new, olds + ["<already exists>"]))
        if collisions:
            print(f"[{exp}] ABORT -- {len(collisions)} name collision(s); nothing renamed:")
            for new, olds in collisions:
                print(f"    {new}")
                for o in olds:
                    print(f"        <- {o}")
            skipped += len(plan)
            continue

        print(f"[{exp}] {len(plan)} to rename, {unchanged} already canonical, "
              f"{baselines} baseline (left alone), {len(unparsed)} unparseable")
        for name in unparsed:
            print(f"    ?? {name}")
        for old, new in plan:
            link = " (symlink)" if os.path.islink(os.path.join(root, old)) else ""
            if args.apply:
                os.rename(os.path.join(root, old), os.path.join(root, new))
                print(f"    RENAMED{link} {old}\n         -> {new}")
                renamed += 1
            else:
                print(f"    would rename{link} {old}\n              -> {new}")

    print("-" * 62)
    if args.apply:
        print(f"renamed {renamed} folder(s) out of {total} inspected"
              + (f"; {skipped} skipped due to collisions" if skipped else ""))
    else:
        print(f"{total} folder(s) inspected. Re-run with --apply to rename.")


if __name__ == "__main__":
    main()
