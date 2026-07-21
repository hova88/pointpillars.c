"use strict";

const $ = (selector) => document.querySelector(selector);
const formatCount = (value) => new Intl.NumberFormat("en-US").format(value);
const formatBytes = (value, unit) => `${(value / (unit === "MiB" ? 2 ** 20 : 2 ** 10)).toFixed(2)} ${unit}`;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderFlow(flow) {
  const rail = $("#flow-rail");
  const show = (index) => {
    const item = flow[index];
    rail.querySelectorAll("button").forEach((button, current) => {
      button.setAttribute("aria-selected", String(current === index));
      button.tabIndex = current === index ? 0 : -1;
    });
    $("#flow-index").textContent = `${String(index + 1).padStart(2, "0")} / ${String(flow.length).padStart(2, "0")}`;
    $("#flow-name").textContent = item.operator;
    $("#flow-shape").textContent = item.shape;
    $("#flow-note").textContent = item.note;
  };

  flow.forEach((item, index) => {
    const button = element("button");
    button.type = "button";
    button.setAttribute("role", "tab");
    button.append(element("span", "", String(index + 1).padStart(2, "0")), element("b", "", item.operator));
    button.addEventListener("click", () => show(index));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let next = index + (event.key === "ArrowDown" ? 1 : -1);
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = flow.length - 1;
      next = (next + flow.length) % flow.length;
      show(next);
      rail.children[next].focus();
    });
    rail.append(button);
  });
  show(0);
}

function renderComposition(groups) {
  const host = $("#operator-composition");
  const learned = groups.filter((group) => group.id !== "metadata");
  const max = Math.max(...learned.map((group) => group.modules));
  learned.forEach((group) => {
    const row = element("div", "composition-row");
    const track = element("span", "composition-bar");
    const bar = element("i");
    bar.style.setProperty("--w", `${Math.max(1.5, group.modules / max * 100)}%`);
    track.append(bar);
    row.append(element("span", "", group.title), track,
      element("b", "", `${group.modules} · ${group.gmac.toFixed(3)}G`));
    host.append(row);
  });
}

function renderModelBrowser(data) {
  $("#stat-tensors").textContent = formatCount(data.model.tensors);
  $("#stat-modules").textContent = formatCount(data.summary.modules);
  $("#stat-operators").textContent = formatCount(data.summary.learned_operators);
  $("#stat-elements").textContent = `${(data.model.elements / 1e6).toFixed(2)} M`;
  renderComposition(data.groups);

  const filters = $("#group-filters");
  const search = $("#module-search");
  const list = $("#module-list");
  const empty = $("#module-empty");
  const titles = Object.fromEntries(data.groups.map((group) => [group.id, group.title]));
  let active = "all";
  let expanded = false;

  [{id: "all", title: "All"}, ...data.groups].forEach((group) => {
    const button = element("button", "", group.title);
    button.type = "button";
    button.dataset.group = group.id;
    button.setAttribute("aria-pressed", String(group.id === active));
    button.addEventListener("click", () => {
      active = group.id;
      expanded = false;
      filters.querySelectorAll("button").forEach((item) =>
        item.setAttribute("aria-pressed", String(item.dataset.group === active)));
      render();
    });
    filters.append(button);
  });

  function render() {
    const query = search.value.trim().toLowerCase();
    const matches = data.modules.filter((module) =>
      (active === "all" || module.group === active) &&
      (!query || `${module.name} ${module.operator} ${titles[module.group]}`.toLowerCase().includes(query)));
    const visible = expanded ? matches : matches.slice(0, 24);
    list.replaceChildren();
    empty.hidden = matches.length !== 0;
    $("#browser-summary").replaceChildren(
      element("strong", "", `${matches.length} stored ${matches.length === 1 ? "entry" : "entries"}`),
      element("span", "", `${formatCount(matches.reduce((sum, item) => sum + item.elements, 0))} fp32 elements`));

    visible.forEach((module) => {
      const details = element("details", "module-row");
      const summary = element("summary");
      const shapeText = module.tensor_shapes.map((shape) => `[${shape.join(" × ")}]`).join(" · ");
      summary.append(element("span", "op-pill", module.operator),
        element("code", "module-name", module.name),
        element("code", "module-shape", shapeText),
        element("span", "module-count", formatCount(module.elements)));
      const body = element("div", "module-detail");
      body.append(element("p", "", `${titles[module.group]} · ${formatBytes(module.bytes, "KiB")}`));
      module.tensor_names.forEach((name, index) => {
        const row = element("div");
        row.append(element("code", "", name), element("span", "", `[${module.tensor_shapes[index].join(", ")}]`));
        body.append(row);
      });
      details.append(summary, body);
      list.append(details);
    });

    if (!expanded && matches.length > visible.length) {
      const more = element("button", "show-more", `Show all ${matches.length} entries`);
      more.type = "button";
      more.addEventListener("click", () => { expanded = true; render(); });
      list.append(more);
    }
  }
  search.addEventListener("input", () => { expanded = false; render(); });
  render();
}

const performanceCaptions = {
  cpu: "Strict C/OpenMP/AVX2 path; the checkpoint oracle passes at 1.0023e-3 maximum absolute error.",
  cuda_raw: "Fast custom FP16/WMMA path with the full raw tensor boundary; approximate, not a graph-equivalence claim.",
  cuda_compact: "Fast custom FP16/WMMA path with device-side candidate filtering; approximate, with canonical CPU decode and NMS.",
  cudnn_raw: "Strict FP32/FMA cuDNN convolutions; all 3.87M raw floats return to the host and pass the oracle at 4.9973e-4.",
  cudnn_compact: "Strict FP32/FMA cuDNN convolution; compact candidates return to the canonical CPU decoder and NMS."
};

function renderPerformance(performance) {
  const reports = performance.reports;
  if (!reports.length) return;
  const tabs = $("#perf-tabs");
  const stageColors = {pfn: "#f59e6b", scatter: "#2bb7ad", backbone: "#1769e0", heads: "#78a9e8"};
  const stageLabels = {pfn: "PFN", scatter: "Scatter", backbone: "Backbone", heads: "Heads"};
  const cpu = reports.find((report) => report.id === "cpu") || reports[0];

  const show = (report) => {
    tabs.querySelectorAll("button").forEach((button) => {
      const selected = button.dataset.report === report.id;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    $("#perf-label").textContent = report.name;
    $("#perf-total").textContent = `${report.warm_median_ms.toFixed(3)} ms`;
    $("#perf-cold").textContent = `${report.cold_ms.toFixed(3)} ms`;
    $("#perf-p95").textContent = `${report.warm_p95_ms.toFixed(3)} ms`;
    $("#perf-workspace").textContent = formatBytes(report.workspace_bytes, "MiB");
    $("#perf-d2h").textContent = report.device_to_host_bytes ? formatBytes(report.device_to_host_bytes, "KiB") : "none";
    $("#perf-caption").textContent = performanceCaptions[report.id] || "Measured end-to-end on the named fixture.";
    const chart = $("#stage-chart");
    chart.replaceChildren();
    const stageMax = Math.max(...Object.values(report.stages_ms));
    Object.entries(report.stages_ms).forEach(([name, value]) => {
      const row = element("div", "stage-row");
      const track = element("span", "stage-track");
      const bar = element("i");
      bar.style.setProperty("--w", `${value ? Math.max(1, value / stageMax * 100) : 0}%`);
      bar.style.setProperty("--c", stageColors[name] || "#9aa7b4");
      track.append(bar);
      row.append(element("span", "", stageLabels[name] || name), track,
        element("span", "", `${value.toFixed(3)} ms`));
      chart.append(row);
    });
  };

  reports.forEach((report, index) => {
    const button = element("button", "", report.name.replace(" · ", " / "));
    button.type = "button";
    button.dataset.report = report.id;
    button.setAttribute("role", "tab");
    button.addEventListener("click", () => show(report));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let next = index + (event.key === "ArrowRight" ? 1 : -1);
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = reports.length - 1;
      next = (next + reports.length) % reports.length;
      show(reports[next]);
      tabs.children[next].focus();
    });
    tabs.append(button);
  });

  const ladder = $("#performance-ladder");
  reports.forEach((report) => {
    const row = element("div", `ladder-row${report.warm_median_ms === Math.min(...reports.map((item) => item.warm_median_ms)) ? " best" : ""}`);
    const track = element("span", "ladder-track");
    const bar = element("i");
    bar.style.setProperty("--w", `${Math.max(1.5, report.warm_median_ms / cpu.warm_median_ms * 100)}%`);
    track.append(bar);
    row.append(element("span", "", report.name), track, element("b", "", `${report.warm_median_ms.toFixed(3)} ms`));
    ladder.append(row);
  });
  show(reports.find((report) => report.id === "cudnn_compact") || reports[0]);
}

async function boot() {
  try {
    const response = await fetch("model-data.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderFlow(data.flow);
    renderModelBrowser(data);
    renderPerformance(data.performance);
  } catch (error) {
    document.body.classList.add("data-error");
    console.error("Unable to load PointPillars site data", error);
  }
}

boot();
