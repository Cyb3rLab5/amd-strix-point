## 2024-05-24 - Exception-based performance bottleneck in loops
**Learning:** Calling `get_cuda_free_memory_gb` inside tight loops like `model.modules()` causes a massive performance bottleneck on non-CUDA devices because the fallback logic relies on throwing and catching an exception on every call.
**Action:** When a function relies on exception handling for standard logic (like detecting memory availability), avoid placing it inside a loop, or implement an early-exit if the parameter conditions (`preserved_memory_gb <= 0`) make the check unnecessary.
