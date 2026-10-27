# Reinforcement Learning from Human Feedback (RLHF)

RLHF is the technique that turns a pretrained, next-token-prediction
language model into one that follows instructions and produces outputs
humans actually prefer. Pretraining alone optimizes for predicting
plausible text continuations, which isn't the same objective as "be a
helpful, honest assistant" — RLHF closes that gap using human preference
data as the training signal, since "helpfulness" isn't something you can
write a differentiable loss function for directly.

## The three-stage pipeline

**Supervised fine-tuning (SFT):** the pretrained model is fine-tuned on a
smaller dataset of high-quality demonstration examples — prompts paired
with the response a human would want — to shift its behavior toward
instruction-following before any reward modeling happens.

**Reward model training:** human labelers are shown multiple model outputs
for the same prompt and rank them by preference. A separate reward model
is trained to predict these preference judgments — given a prompt and a
response, output a scalar score such that preferred responses score
higher. This reward model becomes a learned proxy for "how good is this
response," standing in for the human judgment that can't scale to every
training step.

**RL fine-tuning:** the SFT model is further trained with reinforcement
learning (originally PPO — Proximal Policy Optimization) to maximize the
reward model's score, with a KL-divergence penalty against the original
SFT model to prevent it from drifting so far that it starts exploiting
quirks of the reward model rather than genuinely improving (reward
hacking).

## Why not skip straight to RL

Training the initial SFT stage first matters because starting RL from the
raw pretrained model would mean exploring from a policy that doesn't yet
produce coherent, on-topic responses — the reward model's judgments are
far less reliable on that kind of output, and RL exploration from a bad
starting policy is far less sample-efficient.

## DPO and RLHF alternatives

Direct Preference Optimization (DPO) reformulates the same underlying
objective without an explicit reward model or RL loop: it derives a loss
function directly from the preference data that has the same optimum as
the full RLHF pipeline under certain assumptions, and optimizes it with
standard supervised learning. This is simpler to implement and tune (no
separate reward model, no RL stability issues) and has become a common
default, though the original RLHF/PPO pipeline can still outperform DPO in
some settings, particularly when the preference data itself is noisy or
when online exploration (sampling new responses during training, which
DPO's offline formulation doesn't do) meaningfully helps.

## Reward hacking

The reward model is a proxy, not the true objective, and optimizing hard
against a proxy is a well-known failure mode: a policy can learn to
exploit reward model blind spots (padding responses with unnecessary
length because the reward model happens to correlate length with quality,
for instance) without genuinely improving the property humans actually
care about. This is why the KL penalty against the SFT model exists, and
why reward models themselves need to be evaluated and iterated on, not
treated as a fixed, ground-truth signal.
