## 2024-05-24 - Device type checking in PyTorch
**Learning:** Checking `getattr(device, 'type', '')` is dangerous because PyTorch device arguments can frequently be passed as integers (e.g. `0`) or strings (e.g. `'cuda:0'`), neither of which have a `type` attribute, leading to silent failures when identifying CUDA devices.
**Action:** When validating PyTorch device identifiers, always check `isinstance(device, int) or str(device).startswith('cuda')` to safely handle integers, strings, and torch.device objects.
## 2024-06-03 - PyTorch Tensor Reallocation in Hot Loops
**Learning:** Hardcoded arrays converted to `torch.tensor` inside frequently called functions (like `vae_decode_fake` used for video frame previews) cause significant performance overhead due to constant memory reallocation and CPU-to-GPU transfers.
**Action:** Extract static data to module level and implement a dictionary cache keyed by `(device, dtype)` to reuse tensors. This avoids repeated overhead while supporting dynamic device placement.

## 2024-06-03 - VRAM Polling Overhead in Model Iteration
**Learning:** Calling `torch.cuda.memory_stats()` or `torch.cuda.mem_get_info()` continuously during module iteration (like moving a large model with `model.modules()`) introduces a massive bottleneck. The CUDA runtime syncs the CPU and GPU on each check, causing a massive stall.
**Action:** When tracking VRAM floor/ceiling limits dynamically, always batch the polling checks using a modulus (e.g. `if i % 25 == 0`) rather than checking on every single iteration step.
## 2026-07-06 - NumPy poly1d vs Native Evaluation
**Learning:** Evaluating `np.poly1d` on single scalar values in hot loops introduces significant Numpy overhead (~12us vs ~0.3us for native Python). Horner's method provides the exact mathematical equivalent but is drastically faster when evaluating small polynomials on the CPU.
**Action:** When a static low-degree polynomial needs to be evaluated frequently on a scalar, avoid `np.poly1d` at the call site. Instead, store the coefficients and evaluate using a native Python unrolled Horner's method implementation.
