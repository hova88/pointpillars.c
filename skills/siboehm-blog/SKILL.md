---
name: siboehm-blog
description: Create or revise deeply technical, illustrated systems-performance articles and repository wikis using an evidence-led optimization-worklog structure. Use when Codex must explain C/C++/CUDA runtimes, kernels, cache behavior, memory bandwidth, locality, tensor layouts, execution stages, benchmarks, negative results, or optimization history in publication-ready English Markdown.
---

# Si Boehm Systems Blog

Build an auditable performance narrative, not a source-code tour. Start from a concrete contract, derive physical costs, visualize the data movement, and let measurements decide which implementation wins.

## Required reading

Read these references before drafting:

- [references/narrative-patterns.md](references/narrative-patterns.md) for article structure and voice.
- [references/systems-analysis.md](references/systems-analysis.md) for CPU/GPU analysis.
- [references/visual-language.md](references/visual-language.md) before creating diagrams.

Use [assets/article-template.md](assets/article-template.md) as a starting skeleton when creating a standalone article.

## Workflow

1. Freeze the artifact and evidence.
   - Inventory source, headers, tools, tests, configurations, data, and generated artifacts.
   - Record the exact model/data/hardware pairing for every numerical claim.
   - Separate measured facts, source-derived calculations, and hypotheses.
2. Establish the contract.
   - State shapes, layouts, dtypes, limits, preprocessing, outputs, and failure behavior.
   - Draw the end-to-end execution path before discussing optimizations.
3. Build a cost model.
   - Calculate compulsory bytes, workspace residency, cache-line use, arithmetic intensity, synchronization, and transfer boundaries.
   - Name the likely bottleneck, then seek evidence that can falsify it.
4. Walk the implementation in performance order.
   - Explain each module as a data transformation and ownership boundary.
   - For each hot loop, identify iteration ownership, address progression, reuse distance, vectorization, parallel granularity, and edge behavior.
5. Tell optimization history as a ladder.
   - Give every version a mechanism, expected win, measured result, numerical gate, memory cost, and fallback.
   - Include negative results when they teach a transferable constraint.
6. Create original diagrams.
   - Prefer SVG for repository-native diagrams and Mermaid for simple dependency flows.
   - Show addresses, tiles, cache levels, ownership, or time—not decorative architecture boxes.
7. Publish a navigable document set.
   - Provide a short home page, ordered reading path, cross-links, glossary, reproducible commands, and extension seams.
   - Keep each page useful independently while avoiding duplicated explanations.
8. Validate.
   - Run `python3 scripts/check_wiki.py <wiki-directory>` from this skill.
   - Re-run repository tests and every benchmark quoted in the final copy when feasible.

## Writing constraints

- Write in English unless explicitly asked otherwise.
- Use a curious worklog voice without impersonating or copying any author.
- Prefer exact nouns, short paragraphs, inline equations, annotated snippets, and honest uncertainty.
- Do not claim a bottleneck from latency alone; connect time to bytes, operations, stalls, or synchronization.
- Do not call an optimization successful without before/after measurements and a correctness gate.
- Keep raw source citations clickable and place external references next to the supported claim.
- Never reuse third-party prose or diagrams; derive fresh explanations and visuals from the live code.
