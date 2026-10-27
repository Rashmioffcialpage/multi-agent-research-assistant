# Evaluating LLM Applications

Evaluating an LLM application is harder than evaluating a traditional
classifier because there's usually no single correct output to compare
against — "write a good summary of this document" has many valid answers,
and even the same underlying model will phrase them differently across
runs. Evaluation approaches split roughly into reference-based metrics,
LLM-as-judge, and human evaluation, each with different cost and
reliability tradeoffs.

## Reference-based metrics

Metrics like BLEU and ROUGE (originally built for machine translation and
summarization respectively) score generated text by n-gram overlap against
a reference answer. They're cheap and deterministic, but correlate poorly
with actual quality for open-ended generation — a correct answer phrased
differently from the reference scores low, and a fluent-but-wrong answer
that happens to share vocabulary with the reference can score
deceptively high. They remain useful for tasks with genuinely narrow,
well-defined correct outputs, and much less useful as the sole metric for
free-form generation or RAG answers.

## LLM-as-judge

Using a (usually stronger) LLM to score another model's output against a
rubric — faithfulness, relevance, coherence, correctness against a
reference — scales better than human evaluation and correlates
reasonably well with human judgment on many tasks, which is why RAG
evaluation frameworks like RAGAS and DeepEval lean on it heavily: faithfulness
(does the answer only make claims the retrieved context supports) and
answer relevance (does the answer actually address the question, rather
than being generically true) are both naturally phrased as judge prompts.
The real risks are judge bias (LLM judges have been shown to favor
longer answers, and outputs stylistically similar to their own default
style) and cost (a judge call per evaluated example adds up at scale) —
neither disqualifying, but both real enough that judge-based scores should
be spot-checked against human judgment periodically rather than trusted
blindly forever.

## Task-specific automated checks

Some properties don't need a judge at all — they're checkable directly.
Does structured output actually parse as valid JSON against the expected
schema. Does a generated SQL query execute without error and return a
non-empty result. Does a code-generation task's output pass the given unit
tests. These are strictly more reliable than LLM-as-judge where they
apply, precisely because they're not asking a model to judge a model —
prefer them whenever the property being checked has an objective,
programmatic definition.

## Regression testing for prompts

Because prompt changes can improve one case while silently regressing
another, a fixed evaluation set — a curated collection of representative
inputs with either reference answers or a rubric, run after every prompt
or model change — functions the same way a unit test suite does for
traditional code: it catches regressions a single manual spot-check would
miss, and it's what turns "I tweaked the prompt and it felt better" into
an actual, comparable measurement.
