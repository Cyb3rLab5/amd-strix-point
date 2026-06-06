## 2024-06-06 - Memory Verification Bottleneck Optimization
**Learning:** Checking `get_cuda_free_memory_gb` requires internal Torch memory stats fetching or exceptions (such as unsupported CPU stats), which has a noticeable cumulative overhead (~100ms+) when checking thousands of times during a single model device move. Because we only actually move weights, we can drastically reduce checks.
**Action:** When iterating over `model.modules()`, only evaluate memory statistics if `hasattr(m, 'weight')` is True. This eliminates thousands of slow exception-handling fallback calls.
