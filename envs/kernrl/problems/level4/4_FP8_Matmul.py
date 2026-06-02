import torch
import torch.nn as nn

# FP8 Matrix Multiplication with Tensor Cores
# Reference: FP8 Formats for Deep Learning (https://arxiv.org/abs/2209.05433)
#
# FP8 is an 8-bit floating point format used for efficient inference:
# - E4M3: 4 exponent bits, 3 mantissa bits (higher precision, smaller range)
# - E5M2: 5 exponent bits, 2 mantissa bits (lower precision, larger range)
#
# Modern GPUs (H100, B200) have native FP8 tensor cores providing 2x throughput
# over FP16. The challenge is maintaining numerical accuracy with quantization.
#
# This problem tests:
# 1. FP8 quantization (scale computation, clamping)
# 2. FP8 GEMM with tensor cores (torch._scaled_mm)
# 3. Proper scale factor handling
#
# PyTorch 2.1+ supports torch.float8_e4m3fn and torch.float8_e5m2
# torch._scaled_mm provides native FP8 tensor core GEMM


class Model(nn.Module):
    """
    FP8 Matrix Multiplication using torch._scaled_mm for tensor core acceleration.

    This baseline uses the proper FP8 tensor core path:
    - Quantizes inputs/weights to FP8 with per-tensor scaling
    - Uses torch._scaled_mm for actual FP8 tensor core GEMM
    - Achieves ~2x throughput over FP16 on H100/B200

    Key optimization targets for a custom kernel:
    1. Fused quantize-matmul pipeline (avoid separate scale computation)
    2. Per-channel or block-wise scaling for better accuracy
    3. Delayed scaling / amax history for training stability
    4. Memory-efficient weight storage (pre-quantized FP8 weights)

    The baseline implementation:
    - Computes per-tensor scale dynamically
    - Quantizes activations and weights each forward pass
    - Uses torch._scaled_mm for FP8 GEMM

    An optimized kernel could:
    - Pre-quantize weights and store scales
    - Use block-wise scaling for better accuracy
    - Fuse scale computation into the GEMM kernel
    """

    def __init__(self, M: int, K: int, N: int, use_e4m3: bool = True):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        self.use_e4m3 = use_e4m3

        # FP8 format specifications
        if use_e4m3:
            self.fp8_dtype = torch.float8_e4m3fn
            self.fp8_max = 448.0  # Max representable value in E4M3
        else:
            self.fp8_dtype = torch.float8_e5m2
            self.fp8_max = 57344.0  # Max representable value in E5M2

        # Weight matrix stored in FP16 (quantized dynamically in forward)
        # In production, weights would be pre-quantized to FP8
        self.weight = nn.Parameter(torch.randn(K, N) * 0.02)

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-tensor scale for FP8 quantization."""
        amax = x.abs().max()
        scale = self.fp8_max / amax.clamp(min=1e-12)
        return scale

    def quantize_to_fp8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Quantize FP16/BF16 tensor to FP8."""
        x_scaled = x * scale
        x_clamped = x_scaled.clamp(-self.fp8_max, self.fp8_max)
        return x_clamped.to(self.fp8_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FP8 matmul using tensor cores: x @ weight

        Input x: (batch, seq_len, K) in FP16/BF16
        Weight: (K, N) in FP16
        Output: (batch, seq_len, N) in FP16/BF16

        Uses torch._scaled_mm which requires:
        - A: (M, K) in FP8, row-major
        - B: (N, K) in FP8, row-major (transposed internally)
        - scale_a, scale_b: scalar scales (inverse of quantization scale)
        """
        input_dtype = x.dtype
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Reshape for matmul: (batch, seq, K) -> (batch*seq, K)
        x_2d = x.view(-1, self.K)

        # Compute scales for dynamic quantization
        x_scale = self.compute_scale(x_2d)
        w_scale = self.compute_scale(self.weight)

        # Quantize to FP8
        x_fp8 = self.quantize_to_fp8(x_2d, x_scale)

        # For _scaled_mm, weight needs to be (N, K) row-major
        # Original weight is (K, N), so transpose and quantize
        w_t = self.weight.t().contiguous()  # (N, K)
        w_fp8 = self.quantize_to_fp8(w_t, w_scale)

        # Inverse scales for _scaled_mm (it multiplies by these)
        x_scale_inv = (1.0 / x_scale).to(torch.float32)
        w_scale_inv = (1.0 / w_scale).to(torch.float32)

        # FP8 GEMM using tensor cores
        # _scaled_mm computes: (A @ B.T) * scale_a * scale_b
        # A: (M, K), B: (N, K) -> output: (M, N)
        out = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),  # _scaled_mm expects B then transposes it
            scale_a=x_scale_inv,
            scale_b=w_scale_inv,
            out_dtype=input_dtype,
        )

        return out.view(batch_size, seq_len, self.N)


# Configuration sized for H100/B200 tensor cores
batch_size = 8
seq_len = 2048
M = batch_size * seq_len  # Total rows
K = 4096  # Hidden dimension
N = 4096  # Output dimension
use_e4m3 = True  # E4M3 is more common for weights/activations


def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]


def get_init_inputs():
    return [M, K, N, use_e4m3]
