"""RULER version of ``experiments/LongBench/figure_making_script/fig9_budget_equivalence/plot_budget.py`` (plot stage).

The script uses the newest ``.npz`` in this directory and calls the LongBench module.
"""
import os
import sys
LB_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                      "LongBench", "figure_making_script"))
sys.path.insert(0, os.path.join(LB_FIG, "fig9_budget_equivalence"))
HERE = os.path.dirname(os.path.abspath(__file__))
if "--data" not in sys.argv and "--data-dir" not in sys.argv:
    sys.argv += ["--data-dir", HERE]       # png is written next to the chosen .npz
import plot_budget as _mod
_mod.main()
