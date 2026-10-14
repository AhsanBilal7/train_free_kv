"""SnapKV, PyramidKV, StreamingLLM, Ada-KV and CriticalKV baselines and the per-head cache.

Adapted from AdaKV (Feng et al., "Ada-KV: Optimizing KV Cache Eviction by Adaptive
Budget Allocation for Efficient LLM Inference", 2025; https://github.com/FFY0/AdaKV,
MIT). AdaKV contains code from SnapKV (Li et al., NeurIPS 2024;
https://github.com/FasterDecoding/SnapKV) and PyramidKV (Cai et al., 2025;
https://github.com/Zefan-Cai/PyramidKV). The ``critical`` mode adds the CriticalKV
selection (Feng et al., "Identify Critical KV Cache in LLM Inference from an Output
Perturbation Perspective", ICML 2025) to the Ada-KV cluster of this package.
"""
