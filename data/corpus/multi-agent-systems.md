# Multi-Agent LLM Systems

A multi-agent LLM system splits a task across several LLM-driven roles
instead of one model doing everything in a single prompt. Each agent has a
narrower responsibility, its own prompt/context, and sometimes its own
tools, and the system's behavior emerges from how they're composed —
sequential handoffs, a supervisor routing to specialists, or a graph with
loops and conditional branches.

## Why split the task at all

A single, long, do-everything prompt tends to degrade on complex tasks:
instructions compete for the model's attention, intermediate reasoning
gets entangled with final output formatting, and there's no natural point
to insert a check before committing to an answer. Splitting into roles
(e.g., a planner that decomposes the task, a worker that executes each
part, a critic that reviews the result before it's returned) gives each
prompt a single job, makes intermediate state inspectable and testable
independently, and creates a natural place to add a self-correction loop:
if the critic rejects the output, control routes back to the worker with
specific feedback instead of failing outright.

## Orchestration patterns

**Sequential pipeline**: each agent runs once, in a fixed order, passing
its output to the next. Simple, predictable, but can't adapt if an early
step's output turns out to be insufficient.

**Supervisor/router**: a coordinating agent looks at the current state and
decides which specialist agent to invoke next — useful when the right next
step genuinely depends on what's happened so far (e.g., route to a
"needs more research" branch vs. a "ready to answer" branch).

**Graph with cycles**: the most general pattern, and what frameworks like
LangGraph are built around — agents are nodes in a directed graph,
edges can be conditional on the current state, and cycles are allowed
(a critic node can route back to a researcher node). This is what makes
iterative refinement possible: the graph doesn't have to know in advance
how many iterations a given input will need.

## State management

The defining engineering problem in a multi-agent graph is state: what
each node reads, what it writes, and how conflicting or partial updates
from different nodes get merged. LangGraph represents this as a typed
state object threaded through every node, with each node returning a
partial update that gets merged into the running state (by default,
overwrite; reducers can be specified for accumulation instead — e.g., a
list of messages that appends rather than replaces). Getting this contract
right — deciding exactly what belongs in shared state vs. what's
local to one node's prompt — is most of the actual design work in
building a multi-agent system; the individual agent prompts are usually
the easy part.

## Failure modes

Multi-agent systems introduce failure modes a single-prompt system doesn't
have: infinite loops (a critic that never approves), state that grows
unboundedly across iterations, and error compounding (an early agent's
mistake becomes an unquestioned premise for every later agent). Production
systems bound iteration counts explicitly, log the full state trace for
debugging, and often have one agent's job include validating the
*previous* agent's output format before proceeding, not just the content.
