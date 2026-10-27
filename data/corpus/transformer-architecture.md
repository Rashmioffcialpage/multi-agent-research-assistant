# The Transformer Architecture

The transformer, introduced in "Attention Is All You Need" (Vaswani et al.,
2017), replaced recurrence and convolution with self-attention as the core
mechanism for sequence modeling. Every token in a sequence attends directly
to every other token in a single layer, which removes the sequential
bottleneck of RNNs and lets training parallelize across the full sequence
length on a GPU or TPU.

## Self-attention

Self-attention computes three projections of each token's embedding:
a query (Q), a key (K), and a value (V). Attention weights are the scaled
dot product of Q and K, passed through a softmax, then used to compute a
weighted sum of V. Multi-head attention runs several of these attention
computations in parallel with separate learned projections, letting
different heads specialize in different relationships (e.g., one head
tracking syntactic dependencies, another tracking coreference).

## Position information

Because self-attention has no inherent notion of token order, transformers
add positional information explicitly — either fixed sinusoidal position
encodings (the original paper's choice), learned absolute position
embeddings, or, in most modern LLMs, relative schemes like rotary position
embeddings (RoPE), which encode relative position directly into the
attention computation and generalize better to sequence lengths longer
than what the model was trained on.

## Encoder-decoder vs. decoder-only

The original transformer was an encoder-decoder: the encoder processes the
full input with bidirectional attention, and the decoder generates output
autoregressively, attending to both its own previous tokens and the
encoder's output (cross-attention). Most modern generative LLMs (GPT-family,
Llama, Qwen) are decoder-only — a single stack of blocks trained with
causal (left-to-right) attention masking, where each token can only attend
to itself and earlier tokens. This design is simpler and scales well for
next-token-prediction pretraining on large text corpora.

## Why it scales

Transformers scale predictably with parameter count, data, and compute —
the empirical basis for the "scaling laws" work that motivated training
ever-larger models. The main computational cost is that self-attention is
O(n²) in sequence length, which is why long-context models rely on
techniques like sliding-window attention, grouped-query attention (fewer
KV heads than query heads, to shrink the KV cache), or attention variants
like FlashAttention that reduce memory movement without changing the math.
