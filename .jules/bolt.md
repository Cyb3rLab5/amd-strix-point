## 2024-05-24 - Device type checking in PyTorch
**Learning:** Checking `getattr(device, 'type', '')` is dangerous because PyTorch device arguments can frequently be passed as integers (e.g. `0`) or strings (e.g. `'cuda:0'`), neither of which have a `type` attribute, leading to silent failures when identifying CUDA devices.
**Action:** When validating PyTorch device identifiers, always check `isinstance(device, int) or str(device).startswith('cuda')` to safely handle integers, strings, and torch.device objects.
## 2024-06-03 - PyTorch Tensor Reallocation in Hot Loops
**Learning:** Hardcoded arrays converted to `torch.tensor` inside frequently called functions (like `vae_decode_fake` used for video frame previews) cause significant performance overhead due to constant memory reallocation and CPU-to-GPU transfers.
**Action:** Extract static data to module level and implement a dictionary cache keyed by `(device, dtype)` to reuse tensors. This avoids repeated overhead while supporting dynamic device placement.

## 2024-06-03 - VRAM Polling Overhead in Model Iteration
**Learning:** Calling `torch.cuda.memory_stats()` or `torch.cuda.mem_get_info()` continuously during module iteration (like moving a large model with `model.modules()`) introduces a massive bottleneck. The CUDA runtime syncs the CPU and GPU on each check, causing a massive stall.
**Action:** When tracking VRAM floor/ceiling limits dynamically, always batch the polling checks using a modulus (e.g. `if i % 25 == 0`) rather than checking on every single iteration step.
## 2024-06-18 - Avoiding .item() and numpy in Hot GPU Loops
**Learning:** Calling `.item()`, `.tolist()`, or passing GPU tensors into numpy polynomials (like `np.poly1d`) inside hot model execution loops (like the diffusion transformer's TEACache checking logic) triggers severe CPU-GPU synchronization blocks. These blocks stall the GPU pipeline and destroy parallel performance.
**Action:** Replace `np.poly1d` with PyTorch-native Horner's method arithmetic to evaluate polynomials entirely on the GPU. Remove `.item()` calls where possible by leveraging PyTorch's native ability to slice tensors using 0D integer tensors (`tensor[:, :length_tensor]`), keeping operations asynchronous and unblocked. Vectorize sequence length calculations.
