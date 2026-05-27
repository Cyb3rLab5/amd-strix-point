## 2024-05-28 - Optimize Memory Profiling
**Learning:** Checking `torch.cuda.memory_stats()` can be slow when it's done repeatedly during model offloading/moving, because it synchronizes the device and creates a large dictionary of memory statistics.
**Action:** Replaced `torch.cuda.memory_stats()` overhead with direct calls to `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` to fetch just the active and reserved memory, avoiding building a dictionary of the full memory stats.
