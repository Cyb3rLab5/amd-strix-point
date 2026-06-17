## 2024-05-24 - Device type checking in PyTorch
**Learning:** Checking `getattr(device, 'type', '')` is dangerous because PyTorch device arguments can frequently be passed as integers (e.g. `0`) or strings (e.g. `'cuda:0'`), neither of which have a `type` attribute, leading to silent failures when identifying CUDA devices.
**Action:** When validating PyTorch device identifiers, always check `isinstance(device, int) or str(device).startswith('cuda')` to safely handle integers, strings, and torch.device objects.
## 2024-06-03 - PyTorch Tensor Reallocation in Hot Loops
**Learning:** Hardcoded arrays converted to `torch.tensor` inside frequently called functions (like `vae_decode_fake` used for video frame previews) cause significant performance overhead due to constant memory reallocation and CPU-to-GPU transfers.
**Action:** Extract static data to module level and implement a dictionary cache keyed by `(device, dtype)` to reuse tensors. This avoids repeated overhead while supporting dynamic device placement.

## 2024-06-03 - VRAM Polling Overhead in Model Iteration
**Learning:** Calling `torch.cuda.memory_stats()` or `torch.cuda.mem_get_info()` continuously during module iteration (like moving a large model with `model.modules()`) introduces a massive bottleneck. The CUDA runtime syncs the CPU and GPU on each check, causing a massive stall.
**Action:** When tracking VRAM floor/ceiling limits dynamically, always batch the polling checks using a modulus (e.g. `if i % 25 == 0`) rather than checking on every single iteration step.
## 2026-06-17 - Device placement in vectorization
**Learning:** Hardcoded strings like `device="cuda"` are fragile and cause failures when the model is on CPU or another backend like MPS/DirectML. When creating new tensors inside model functions, always inherit the device from existing tensors (e.g. `device = input_tensor.device`).
**Action:** Always verify that newly initialized tensors do not use hardcoded device strings and leverage `.device` from input variables.

## 2026-06-17 - PyTorch slicing triggers sync
**Learning:** Even though replacing `.item()` with a 0d tensor avoids explicit sync, passing a tensor into a slice operation (e.g. `tensor[:, :text_len]` where `text_len` is a 0d tensor) still causes implicit CPU-GPU synchronization under the hood because PyTorch must know the resulting tensor shape on the CPU.
**Action:** When optimizing hot loops to avoid synchronization, be mindful that slicing with dynamic tensor variables will still sync.
