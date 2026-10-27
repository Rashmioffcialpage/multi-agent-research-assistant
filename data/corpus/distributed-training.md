# Distributed Training

Training a large model on a single GPU is limited by two independent
constraints: how much compute one device has (training time) and how much
memory it has (whether the model, its activations, and its optimizer
state even fit). Distributed training splits the work across multiple
devices to relax both constraints, using different strategies depending
on which constraint is actually binding.

## Data parallelism

The simplest strategy: replicate the full model on every device, split
each training batch across devices, run forward/backward independently on
each shard, then average gradients across all devices before the
optimizer step (all-reduce). This scales training throughput well when the
model fits on one device but the desired batch size or dataset is large.
It does nothing for the case where the model itself doesn't fit in one
device's memory — every replica still needs the full model.

## Model and tensor parallelism

When a model is too large for one device's memory, it has to be split
across devices instead of replicated. Pipeline parallelism splits the
model by layer — different devices hold different contiguous blocks of
layers, and activations are passed between devices as data flows through
the pipeline (with micro-batching to keep all devices busy rather than
idling while waiting for the previous stage). Tensor parallelism instead
splits individual layers' weight matrices across devices, so a single
matrix multiplication is computed cooperatively across devices with
communication in the middle of the operation — more communication-heavy
than pipeline parallelism, so it generally needs a fast interconnect
(NVLink) between the devices sharing a tensor-parallel group.

## ZeRO and memory-efficient data parallelism

Standard data parallelism wastes memory: every device stores a full copy
of the optimizer state, gradients, and parameters, even though at any
instant only the shard it's currently working with actually needs to be
local. ZeRO (Zero Redundancy Optimizer, from DeepSpeed) partitions
optimizer state, gradients, and optionally parameters themselves across
data-parallel ranks, gathering only what's needed just-in-time for each
computation and releasing it afterward. This gets much of model
parallelism's memory benefit while keeping data parallelism's simpler
communication pattern, at the cost of more frequent (though smaller)
communication.

## 3D parallelism

Training the largest models combines all three: data parallelism across
groups of devices, tensor parallelism within a tightly-interconnected
group (e.g. within one physical server, over NVLink), and pipeline
parallelism across groups (over slower inter-node networking). Getting
the combination right is largely about matching each parallelism
strategy's communication pattern to the actual hardware topology
available — tensor parallelism where interconnect bandwidth is highest,
pipeline and data parallelism where it's lower.
