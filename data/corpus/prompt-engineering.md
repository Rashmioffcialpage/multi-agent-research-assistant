# Prompt Engineering

Prompt engineering is the practice of designing the input to a language
model to reliably get the behavior you want, without changing the model's
weights. It matters because the same model can produce meaningfully
different quality output depending on how a task is framed, what examples
are given, and how the expected output format is specified — and because
it's far cheaper to iterate on a prompt than to fine-tune a model.

## Zero-shot, few-shot, and instructions

Zero-shot prompting just states the task directly with no examples —
works well for tasks a well-instruction-tuned model has effectively seen
the shape of during its own training. Few-shot prompting includes a small
number of worked examples in the prompt itself, which helps most when the
desired output format or reasoning pattern is unusual enough that a plain
instruction under-specifies it — the examples do the specifying that
words alone don't. Clear, explicit instructions (what to do, what not to
do, the exact output format expected) generally beat vague ones regardless
of shot count; a surprising amount of prompt engineering is just writing
an unambiguous specification, the same skill as writing a clear API
contract.

## Chain-of-thought

Asking a model to reason step by step before giving a final answer
measurably improves performance on tasks requiring multi-step reasoning
(arithmetic, logic, multi-hop questions) — the intermediate reasoning
tokens give the model more computation to work with before committing to
an answer, since a transformer's forward pass has bounded computation per
generated token. This has a real cost: more output tokens means higher
latency and cost per query, so chain-of-thought is a tradeoff to apply
where the task actually needs multi-step reasoning, not a default to
always turn on.

## Structured output

Getting a model to reliably return machine-parseable output (JSON matching
a specific schema, for instance) is a common and practical need — an
agent pipeline that expects a list of sub-questions back can't tolerate a
model that sometimes wraps its answer in explanatory prose. Techniques
range from just instructing the format clearly and providing an example,
to constrained decoding (restricting the model's next-token choices at
each step to only tokens that keep the output valid against a grammar or
JSON schema) which guarantees well-formed output at the cost of some
inference-serving complexity. Defensive parsing on the caller's side —
expect malformed output sometimes, validate before using it downstream —
is standard practice regardless of which generation-side technique is
used.

## Prompt injection

Because a language model can't reliably distinguish between "instructions
from the system/developer" and "text it's processing as data" once
they're concatenated into the same input, prompt injection is a real
security concern for any system where a model processes untrusted text
(web content, user-uploaded documents, retrieved search results) alongside
its actual task instructions. Text embedded in that untrusted content can
attempt to override the original instructions. Mitigations include
clearly delimiting untrusted content in the prompt, treating any
instruction-like text found inside retrieved/untrusted content as
suspicious rather than authoritative, and — for anything consequential —
not letting a model take irreversible actions based on instructions that
originated from untrusted input without a human or a separate check in
the loop.
