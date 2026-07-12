# Visual Language

## Contents

1. Purpose
2. Palette and composition
3. Diagram grammar
4. SVG requirements
5. Figure sequence

## 1. Purpose

Every figure must answer one performance question: where bytes live, which addresses are adjacent, what work a thread owns, which buffers overlap in time, or why an optimization changes reuse.

## 2. Palette and composition

Use an original dark workbench aesthetic inspired by hand-annotated systems notes:

- Canvas: `#0b0d10`
- Primary text: `#e8edf2`
- Muted text/grid: `#637083`
- Cyan/data: `#22d3ee`
- Green/compute or output: `#84cc16`
- Orange/hot path: `#f59e0b`
- Purple/cache or shared state: `#a855f7`
- Red/bottleneck or invalid traffic: `#ef4444`

Leave generous negative space. Prefer a few large shapes over dense corporate flowcharts. Use slightly irregular dashed paths or rounded arrows for a notebook feel while keeping labels typeset and readable.

## 3. Diagram grammar

- Rectangles: owned buffers or tiles.
- Nested rectangles: memory hierarchy or tensor subdivisions.
- Solid arrows: data movement.
- Dashed arrows: reuse, prefetch, or logical association.
- Hatched areas: padding, inactive lanes, or wasted capacity.
- Brackets: byte counts, strides, or lifetimes.
- Small colored squares: threads, lanes, points, or candidates.
- A time axis always runs left-to-right; address axes must be labeled.

Pair overview and zoom figures: first show the whole stage, then magnify the address pattern that explains performance.

## 4. SVG requirements

- Use a descriptive `<title>` and `<desc>`.
- Set a `viewBox`; avoid fixed text that clips when scaled.
- Keep text at least 14 px at the target width.
- Use system sans-serif for prose and monospace for shapes, indices, and equations.
- Include marker-based arrowheads and reusable hatch patterns in `<defs>`.
- Do not embed external fonts, raster images, scripts, or third-party artwork.
- Give each Markdown image meaningful alt text and a one-sentence italic caption.

## 5. Figure sequence

A long systems article typically benefits from:

1. End-to-end pipeline and tensor-shape map.
2. Memory-residency/lifetime diagram.
3. CPU cache-line or loop-order close-up.
4. GPU warp/tile ownership close-up.
5. Timeline showing synchronization and overlap.
6. Optimization ladder chart.
7. Validation funnel from operator fixture to task metric.

Do not force all seven. Create the smallest set that makes the physical mechanisms obvious.
