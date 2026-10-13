"""SnapKV, PyramidKV, StreamingLLM and Ada-KV baselines and the per-head cache.

Adapted from AdaKV (Feng et al., "Ada-KV: Optimizing KV Cache Eviction by Adaptive
Budget Allocation for Efficient LLM Inference", 2025; https://github.com/FFY0/AdaKV,
MIT). AdaKV contains code from SnapKV (Li et al., NeurIPS 2024;
https://github.com/FasterDecoding/SnapKV) and PyramidKV (Cai et al., 2025;
https://github.com/Zefan-Cai/PyramidKV). The ``critical`` mode of this repository
uses the Ada-KV cluster of this package.
"""
