## 2024-05-15 - VRAM Polling Bottleneck in Per-Module Loops
**Learning:** Calling `torch.cuda.memory_stats()` within a per-module loop (e.g. iterating over all layers in a transformer to move it to a device) is a severe performance bottleneck. It allocates a massive dictionary of stats on every single iteration.
**Action:** Always use direct API calls like `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` when polling memory within tight loops instead of `memory_stats()`.

## 2024-05-15 - Redundant Tensor Instantiation in Progress Callbacks
**Learning:** Creating tensors from Python lists (e.g., `torch.tensor(list)`) and transferring them to the GPU during every step of a diffusion generation loop (like inside `vae_decode_fake` used for progress previews) causes measurable CPU/GPU synchronization overhead.
**Action:** Cache frequently used static tensors globally per device and dtype to prevent redundant instantiations in tight loops.
