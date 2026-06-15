## 2025-03-01 - PyTorch Tensor Instantiation overhead
**Learning:** `torch.tensor([val] * size).to(device)` is extremely slow due to intermediate Python list creation and cross-device data loading. `torch.full((size,), val, device=device)` is roughly ~3.5x faster. Also, converting lists of tensors into a tensor via `torch.tensor([tensor1, tensor2])` hits slow fallback paths; `torch.stack` or `torch.cat` should be used instead.
**Action:** Always favor native PyTorch constructors (`torch.full`, `torch.zeros`, `torch.ones`, `torch.arange`, `torch.linspace`, `torch.stack`) over Python list-based tensor instantiation, especially inside core evaluation loops like model inference.
## 2024-06-03 - PyTorch Tensor Reallocation in Hot Loops
**Learning:** Hardcoded arrays converted to `torch.tensor` inside frequently called functions (like `vae_decode_fake` used for video frame previews) cause significant performance overhead due to constant memory reallocation and CPU-to-GPU transfers.
**Action:** Extract static data to module level and implement a dictionary cache keyed by `(device, dtype)` to reuse tensors. This avoids repeated overhead while supporting dynamic device placement.

## 2024-06-03 - VRAM Polling Overhead in Model Iteration
**Learning:** Calling `torch.cuda.memory_stats()` or `torch.cuda.mem_get_info()` continuously during module iteration (like moving a large model with `model.modules()`) introduces a massive bottleneck. The CUDA runtime syncs the CPU and GPU on each check, causing a massive stall.
**Action:** When tracking VRAM floor/ceiling limits dynamically, always batch the polling checks using a modulus (e.g. `if i % 25 == 0`) rather than checking on every single iteration step.
