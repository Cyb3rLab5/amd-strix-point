## 2024-05-24 - NumPy poly1d overhead in hot loops
**Learning:** Evaluating `np.poly1d` on single scalar values in PyTorch hot loops introduces significant Numpy overhead (~29x slower than native Python for scalar values).
**Action:** Replace `np.poly1d` evaluations with native Python implementations like Horner's method using standard for loops when evaluating on single scalars extracted via `.cpu().item()`.
