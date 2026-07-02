## 2024-05-24 - Device type checking in PyTorch
**Learning:** Checking `getattr(device, 'type', '')` is dangerous because PyTorch device arguments can frequently be passed as integers (e.g. `0`) or strings (e.g. `'cuda:0'`), neither of which have a `type` attribute, leading to silent failures when identifying CUDA devices.
**Action:** When validating PyTorch device identifiers, always check `isinstance(device, int) or str(device).startswith('cuda')` to safely handle integers, strings, and torch.device objects.
## 2024-06-03 - PyTorch Tensor Reallocation in Hot Loops
**Learning:** Hardcoded arrays converted to `torch.tensor` inside frequently called functions (like `vae_decode_fake` used for video frame previews) cause significant performance overhead due to constant memory reallocation and CPU-to-GPU transfers.
**Action:** Extract static data to module level and implement a dictionary cache keyed by `(device, dtype)` to reuse tensors. This avoids repeated overhead while supporting dynamic device placement.

## 2024-06-03 - VRAM Polling Overhead in Model Iteration
**Learning:** Calling `torch.cuda.memory_stats()` or `torch.cuda.mem_get_info()` continuously during module iteration (like moving a large model with `model.modules()`) introduces a massive bottleneck. The CUDA runtime syncs the CPU and GPU on each check, causing a massive stall.
**Action:** When tracking VRAM floor/ceiling limits dynamically, always batch the polling checks using a modulus (e.g. `if i % 25 == 0`) rather than checking on every single iteration step.
## 2024-05-24 - Numpy overhead in hot loops
**Learning:** `np.poly1d` has significant overhead (around ~1.2s for 100k evals) compared to native python unrolled math (~0.01s for 100k evals) due to function calls, array type casting, and general numpy slowness for single scalar evaluations.
**Action:** When evaluating low-degree polynomials on scalars in hot loops (like in the transformer block's `rel_l1_distance` calculation), always extract the scalar out and use an unrolled Horner's method (`c4 + x*(c3 + x*(c2 + x*(c1 + x*c0)))`) natively instead of `np.poly1d`.
