
## 2025-06-10 - Avoiding 3D Meshgrid Allocations for Coordinate Encoding
**Learning:** In video-oriented models using Rotary Positional Embeddings (RoPE), computing the 3D position coordinates using `torch.meshgrid` natively over `T, H, W` domains redundantly computes trigonometric functions (sin/cos) and allocates massive Cartesian product grids (taking huge amounts of memory and latency per-item in the batch).
**Action:** Always compute coordinate/positional frequencies in independent 1D domains first, shape them properly using `.view()`, and virtually project them into 3D/ND space using PyTorch's zero-copy `.expand()` and broadcasting. This dramatically speeds up coordinate encoding steps.
