## 2025-05-18 - [Optimized Memory Preservation Checks]
**Learning:** Querying `torch.cuda.memory_stats()` via `get_cuda_free_memory_gb()` on *every single module* iteration causes severe blocking overhead during model load/swaps. Large architectures like Llama have >10,000 submodules, making the memory query loop a massive bottleneck.
**Action:** When dynamically moving submodules, only check VRAM thresholds every N steps (e.g. 64) rather than continuously.
