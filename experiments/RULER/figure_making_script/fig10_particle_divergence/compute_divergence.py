"""RULER fig10_particle_divergence - compute stage (thin wrapper).

Reuses experiments/LongBench/figure_making_script/fig10_particle_divergence/compute_divergence.py
with RULER defaults (--benchmark ruler, --dataset 8192). The tagged .npz (config/model/
budget/dataset in its name) is written INTO this folder via --out-dir. All underlying
flags pass through, e.g.: python compute_divergence.py --dataset 4096,8192 --budget 128
"""
import os
import sys
LB_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                      "LongBench", "figure_making_script"))
sys.path.insert(0, LB_FIG)
sys.path.insert(0, os.path.join(LB_FIG, "fig10_particle_divergence"))
HERE = os.path.dirname(os.path.abspath(__file__))
if "--benchmark" not in sys.argv:
    sys.argv += ["--benchmark", "ruler"]
if "--dataset" not in sys.argv:
    sys.argv += ["--dataset", "8192"]
if "--out" not in sys.argv and "--out-dir" not in sys.argv:
    sys.argv += ["--out-dir", HERE]        # tagged .npz lands here
import compute_divergence as _mod
_mod.main()
