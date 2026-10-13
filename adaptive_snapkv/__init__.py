"""📐 SnapKV / PyramidKV / StreamingLLM / Ada-KV baselines and the varlen per-head cache.

Adapted from the AdaKV code base (Feng et al., "Ada-KV: Optimizing KV Cache Eviction
by Adaptive Budget Allocation for Efficient LLM Inference", 2025;
https://github.com/FFY0/AdaKV, MIT), which builds on SnapKV (Li et al., NeurIPS 2024;
https://github.com/FasterDecoding/SnapKV) and PyramidKV (Cai et al., 2025;
https://github.com/Zefan-Cai/PyramidKV). The repository's ``critical`` mode routes
through the Ada-KV cluster defined here.
"""
