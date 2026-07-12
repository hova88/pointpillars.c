# Narrative Patterns

## Contents

1. The worklog arc
2. Page anatomy
3. Voice and pacing
4. Tables, code, and sidenotes
5. Anti-patterns

## 1. The worklog arc

The reference pattern is an optimization diary with a rigorous spine:

1. Open with a specific question and a measurable target.
2. Show a compact performance ladder early so readers know where the story goes.
3. Implement or explain the simplest correct path.
4. Derive a lower bound in bytes, operations, or latency.
5. Visualize why the current access pattern misses that bound.
6. Change one mechanism at a time.
7. Measure, explain the delta, and preserve surprising or negative results.
8. End with remaining gaps and shape-dependent caveats.

The inspiration is Simon Boehm's CUDA matmul worklog: <https://siboehm.com/articles/22/CUDA-MMM>. Extract its teaching mechanics, not its phrasing.

## 2. Page anatomy

Use this repeating unit for a deep section:

> **Question.** What resource is this stage consuming?
>
> **Napkin math.** How many bytes, operations, cache lines, warps, or barriers are unavoidable?
>
> **Picture.** Which addresses or tiles are touched, and by whom?
>
> **Code.** Show only the loop or ownership boundary that makes the mechanism concrete.
>
> **Measurement.** Compare before and after under a named fixture.
>
> **Interpretation.** Explain why the result supports or contradicts the hypothesis.

Start major pages with a one-paragraph outcome and a small fact table. End with "What to remember" and "What remains".

## 3. Voice and pacing

- Sound like an engineer reasoning in public: precise, curious, occasionally surprised.
- Use first person sparingly for experiments: "I expected...; the measurement showed..."
- Put definitions immediately before they become useful.
- Alternate prose, diagram, code, and arithmetic. Avoid walls of any one medium.
- Use parenthetical clarifications and compact sidenotes for real caveats, not trivia.
- State uncertainty: "This is consistent with..." is better than invented certainty.

## 4. Tables, code, and sidenotes

Performance ladders should include mechanism and tradeoff, not only latency:

| Version | Mechanism | Warm latency | Extra memory | Correctness |
|---|---|---:|---:|---|

Code excerpts should be small enough that the access pattern fits on screen. Annotate the changing index or pointer in comments.

Use Markdown blockquotes for sidenotes:

> **Sidenote — shape matters.** A kernel that wins at 256 channels may lose for a 6-channel output head because launch and padding costs dominate.

## 5. Anti-patterns

- File-by-file paraphrase without a data-flow thesis.
- "Cache-friendly" without identifying cache lines, strides, or reuse.
- Peak FLOP comparisons without arithmetic intensity or instruction mix.
- Benchmark tables without hardware, fixture, warmup, and correctness.
- Decorative diagrams that could be replaced by the same list of boxes.
- Hiding regressions, discarded experiments, portability costs, or fallback behavior.
