## 2024-05-24 - Tensor padding optimization
**Learning:** In `diffusers_helper/utils.py`, `add_tensors_with_padding` uses `torch.zeros()` with no dtype or device matching, and assigns tensors which involves large memory allocation twice.
**Action:** Use an in-place addition or single pre-allocation.
