"""RULER version of ``experiments/LongBench/figure_making_script/fig4_entropy_trajectory/compute_entropy.py`` (compute stage).

The script calls the LongBench module with ``--benchmark ruler``, the default
dataset ``8192`` (use ``--dataset 4096`` for the paper setting) and this
directory as the output directory.
"""
import os
import sys
LB_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                      "LongBench", "figure_making_script"))
sys.path.insert(0, LB_FIG)
sys.path.insert(0, os.path.join(LB_FIG, "fig4_entropy_trajectory"))
HERE = os.path.dirname(os.path.abspath(__file__))
if "--benchmark" not in sys.argv:
    sys.argv += ["--benchmark", "ruler"]
if "--dataset" not in sys.argv:
    sys.argv += ["--dataset", "8192"]
if "--out" not in sys.argv and "--out-dir" not in sys.argv:
    sys.argv += ["--out-dir", HERE]        # tagged .npz lands here
import compute_entropy as _mod
_mod.main()
