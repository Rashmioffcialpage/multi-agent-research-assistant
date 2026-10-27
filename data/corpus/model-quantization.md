# Model Quantization

Quantization reduces the numerical precision used to store and compute a
model's weights (and sometimes activations) — from 32-bit floating point
(FP32) down to FP16/BF16, INT8, or INT4 — to shrink model size and, in the
right conditions, speed up inference. A weight matrix stored in INT8
instead of FP32 takes a quarter of the memory, which matters both for
fitting a model in limited RAM/VRAM and for memory-bandwidth-bound
inference, where moving weights from memory to compute units is the
bottleneck, not the arithmetic itself.

## Where the speedup actually comes from

It's a common misconception that lower precision is faster purely because
"smaller numbers are faster to compute." On modern hardware, the real
source of speedup is usually memory bandwidth: LLM inference at small
batch sizes is memory-bound, not compute-bound, so halving or quartering
the bytes that need to move from memory to the arithmetic units directly
reduces latency. This also explains why quantization benefits shrink or
disappear for very small models or very small batches — a tiny model's
weights may already fit comfortably in cache, and the fixed overhead of
quantizing/dequantizing per call can outweigh the reduced data movement
when there isn't much data to move in the first place.

## Quantization-aware training vs. post-training quantization

Post-training quantization (PTQ) takes an already-trained FP32/FP16 model
and converts it directly, optionally using a small calibration dataset to
choose good scale factors per layer. It's cheap — no retraining — but can
lose more accuracy, especially at very low bit widths, because the model
was never trained to be robust to quantization error. Quantization-aware
training (QAT) simulates quantization *during* training (fake-quantizing
weights/activations in the forward pass while keeping full-precision
gradients), so the model learns weights that are robust to the precision
loss it will see at inference time. QAT typically preserves accuracy
better at INT4 and below, at the cost of needing to retrain.

## Dynamic vs. static quantization

Dynamic quantization computes activation quantization parameters on the
fly, per batch, at inference time — no calibration step needed, but some
runtime overhead. Static quantization pre-computes activation ranges from
a calibration dataset ahead of time, so there's no runtime overhead, but
it requires that calibration step and can degrade if real inputs drift
from the calibration distribution.

## INT4 and beyond

Sub-8-bit quantization (INT4, and more exotic schemes like GPTQ and AWQ,
which choose which weights to quantize more/less aggressively based on
their sensitivity) is where accuracy loss becomes a real risk without
careful technique, and where hardware/kernel support is least mature —
many INT4 kernels only exist for GPU, not CPU, which is a genuine
practical constraint when benchmarking on CPU-only hardware: reporting an
INT4 number without a real kernel to produce it honestly would just be
fabricating a result.
