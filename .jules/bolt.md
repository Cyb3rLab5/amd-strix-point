## 2024-05-18 - [Tensor Allocation Overhead]
**Learning:** Re-instantiating small `torch.tensor` variables inside frequently called loops (such as VAE decoding during diffusion preview callbacks) creates measurable performance degradation due to CPU overhead and memory reallocation, even for small lists of floats.
**Action:** Always cache immutable tensors globally (or at a higher scope) if a function is expected to be called rapidly in a tight loop, ensuring they are only updated if `dtype` or `device` changes.
