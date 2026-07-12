# {{Title}}

> **Outcome.** {{One paragraph stating what was built, measured, or learned.}}

| Fixture | Hardware | Baseline | Final | Correctness gate |
|---|---|---:|---:|---|
| {{fixture}} | {{hardware}} | {{baseline}} | {{final}} | {{gate}} |

## The question

{{Define the concrete systems question and why the answer is not obvious.}}

## Contract before optimization

{{Shapes, layouts, dtypes, bounds, preprocessing, and outputs.}}

![{{Descriptive alt text}}](assets/{{figure}}.svg)

*{{Caption explaining what performance fact the reader should see.}}*

## A lower bound

{{Calculate compulsory bytes, operations, and the fastest plausible runtime.}}

## The simple path

```c
/* Small excerpt that exposes ownership and address progression. */
```

{{Explain loop order, locality, cache behavior, and scheduling.}}

## Optimization ladder

| Step | Mechanism | Latency | Memory | Numerical result |
|---|---|---:|---:|---|
| 0 | {{baseline}} | | | |

## {{Optimization 1: mechanism, not marketing name}}

{{Question → napkin math → picture → code → measurement → interpretation.}}

> **Sidenote — {{caveat}}.** {{A useful shape, compiler, or portability caveat.}}

## A result that did not work

{{Record a falsified hypothesis and why it remains useful.}}

## End-to-end accounting

{{Cold/warm latency, throughput, memory residency, transfers, and overlap.}}

## Correctness ladder

{{Operator, graph, decode/NMS, sanitizer, malformed input, and dataset metric evidence.}}

## What to remember

- {{Mechanism-level lesson.}}
- {{Measurement-level lesson.}}
- {{Portability or extension lesson.}}

## What remains

{{Specific unproven hypotheses and the evidence needed next.}}
