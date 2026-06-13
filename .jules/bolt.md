## 2024-06-03 - PyTorch Tensor Reallocation in Hot Loops
**Learning:** Hardcoded arrays converted to `torch.tensor` inside frequently called functions (like `vae_decode_fake` used for video frame previews) cause significant performance overhead due to constant memory reallocation and CPU-to-GPU transfers.
**Action:** Extract static data to module level and implement a dictionary cache keyed by `(device, dtype)` to reuse tensors. This avoids repeated overhead while supporting dynamic device placement.

## 2024-06-03 - PyTorch Optimization Overhead with Python Lists
**Learning:** In C++ natively, python lists of tensor elements suffer from heavy conversion overheads when recreated inside C++ extension functions. Methods like `torch.stack()` significantly outperform `torch.tensor(list)` inside of hot loops by preserving the native tensor structure natively.
**Action:** Replace `torch.tensor` calls with `torch.stack` or class-level constants when aggregating lists of tensors during generation step loops.
