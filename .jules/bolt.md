## 2024-05-24 - Device type checking in PyTorch
**Learning:** Checking `getattr(device, 'type', '')` is dangerous because PyTorch device arguments can frequently be passed as integers (e.g. `0`) or strings (e.g. `'cuda:0'`), neither of which have a `type` attribute, leading to silent failures when identifying CUDA devices.
**Action:** When validating PyTorch device identifiers, always check `isinstance(device, int) or str(device).startswith('cuda')` to safely handle integers, strings, and torch.device objects.
## 2024-06-03 - PyTorch Tensor Reallocation in Hot Loops
**Learning:** Hardcoded arrays converted to `torch.tensor` inside frequently called functions (like `vae_decode_fake` used for video frame previews) cause significant performance overhead due to constant memory reallocation and CPU-to-GPU transfers.
**Action:** Extract static data to module level and implement a dictionary cache keyed by `(device, dtype)` to reuse tensors. This avoids repeated overhead while supporting dynamic device placement.

## 2024-06-03 - VRAM Polling Overhead in Model Iteration
**Learning:** Calling `torch.cuda.memory_stats()` or `torch.cuda.mem_get_info()` continuously during module iteration (like moving a large model with `model.modules()`) introduces a massive bottleneck. The CUDA runtime syncs the CPU and GPU on each check, causing a massive stall.
**Action:** When tracking VRAM floor/ceiling limits dynamically, always batch the polling checks using a modulus (e.g. `if i % 25 == 0`) rather than checking on every single iteration step.

## 2024-06-03 - NumPy poly1d vs Native Horner's Method
**Learning:** In PyTorch hot loops (like within diffusion model transformers),  carries heavy Python overhead for type checking and input parsing. Calling  on scalar values can be an order of magnitude slower than doing the same math explicitly in native Python.
**Action:** Use an unrolled native Python lambda function evaluating with Horner's method to resolve polynomial calculations for scalar inputs in highly iterative paths.

## 2024-06-03 - NumPy poly1d vs Native Horner's Method
**Learning:** In PyTorch hot loops (like within diffusion model transformers), `np.poly1d` carries heavy Python overhead for type checking and input parsing. Calling `np.poly1d` on scalar values can be an order of magnitude slower than doing the same math explicitly in native Python.
**Action:** Use an unrolled native Python lambda function evaluating with Horner's method to resolve polynomial calculations for scalar inputs in highly iterative paths.
