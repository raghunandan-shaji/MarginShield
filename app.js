const state = {
  data: null,
  selected: null,
  filter: "all",
  search: "",
  reversed: false,
  threshold: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const fmtInr = (value) => `INR ${Number(value).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
const pct = (value) => `${Math.round(Number(value) * 100)}%`;
const fmtRisk = (value) => {
  const percent = Number(value);
  if (percent >= 99.95) return ">99.9%";
  if (percent < 0.05) return "<0.1%";
  if (percent >= 99) return `${percent.toFixed(1)}%`;
  if (percent < 1) return `${percent.toFixed(2)}%`;
  return percent < 10 ? `${percent.toFixed(1)}%` : `${Math.round(percent)}%`;
};
const fmtTimestamp = (value) => new Date(value).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
const cssClass = (text) => String(text).toLowerCase().replaceAll(" ", "-");
const esc = (value) => String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2400);
}

function visibleCases() {
  const query = state.search.trim().toLowerCase();
  let cases = state.data.cases.filter((item) => {
    const filterMatch = state.filter === "all" || item.action === state.filter;
    const textMatch = !query || [item.case_id, item.merchant, item.customer, item.reason].join(" ").toLowerCase().includes(query);
    return filterMatch && textMatch;
  });
  if (state.reversed) cases = [...cases].reverse();
  return cases;
}

function renderSummary() {
  const summary = state.data.summary;
  $("#refundExposure").textContent = fmtInr(summary.queue_refund_exposure);
  $("#flaggedLoss").textContent = fmtInr(summary.flagged_loss_exposure);
  $("#highRiskCases").textContent = summary.verify_evidence_cases;
  $("#activeThreshold").textContent = fmtRisk(state.data.active_threshold_percent);
  $("#lastScored").textContent = fmtTimestamp(state.data.as_of);
}

function renderQueue() {
  const cases = visibleCases();
  $("#queueCount").textContent = cases.length;
  const list = $("#caseList");
  list.innerHTML = "";

  if (!cases.length) {
    list.innerHTML = '<p class="empty-queue">No cases match this view. Clear the search or choose another queue.</p>';
    return;
  }

  cases.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `case-card ${state.selected?.case_id === item.case_id ? "is-active" : ""}`;
    const riskClass = item.action === "Verify evidence" ? "high" : item.action === "Manual review" ? "medium" : "";
    button.innerHTML = `
      <span class="case-identity"><strong>${esc(item.case_id)}</strong><span>${esc(item.merchant)} / ${esc(item.reason)}</span></span>
      <span class="case-money">${fmtInr(item.refund_amount).replace("INR ", "")}</span>
      <span class="case-score ${riskClass}">${fmtRisk(item.risk_percent)}</span>
    `;
    button.addEventListener("click", () => {
      state.selected = item;
      renderCase();
      renderQueue();
    });
    list.appendChild(button);
  });
}

function renderDecision() {
  const item = state.selected;
  $("#caseMerchant").textContent = `${item.merchant} / ${item.vertical}`;
  $("#caseTitle").textContent = `${item.case_id} / ${item.customer}`;
  $("#scoreText").textContent = fmtRisk(item.risk_percent);
  $("#scoreMarker").style.left = `calc(${item.risk_percent}% - 1px)`;
  $("#confidenceText").textContent = "Calibrated coordinated-ring probability";
  $("#caseStatus").textContent = item.status;
  $("#caseStatus").className = `risk-label ${cssClass(item.status)}`;
  $("#caseReason").textContent = item.reason;
  $("#actionText").textContent = item.action;
  $("#caseNotes").textContent = item.notes;
  $("#refundAmount").textContent = fmtInr(item.refund_amount);
  $("#expectedLoss").textContent = fmtInr(item.conditional_loss_if_ring);
  $("#fpCost").textContent = fmtInr(item.estimated_false_positive_cost);
  $("#refundMethod").textContent = item.refund_method;
  $("#caseAge").textContent = `Scored ${fmtTimestamp(item.event_timestamp)}`;
}

function renderEvidence() {
  const list = $("#evidenceList");
  const maxContribution = Math.max(...state.selected.evidence.map((item) => Math.abs(item.contribution)), 1);
  list.innerHTML = state.selected.evidence.map((evidence) => {
    const width = Math.min(48, (Math.abs(evidence.contribution) / maxContribution) * 48);
    const direction = evidence.direction === "lowers" ? "relief" : "risk";
    const color = direction === "relief" ? "var(--green)" : "var(--risk)";
    const contribution = Number(evidence.contribution);
    return `
      <div class="evidence-item" title="Directional contribution to model log-odds">
        <span class="evidence-name"><strong>${esc(evidence.name)}</strong><small>Observed: ${esc(evidence.observed)}</small></span>
        <div class="impact-track"><i class="${direction}" style="width:${width}%"></i></div>
        <span style="color:${color}">${contribution > 0 ? "+" : ""}${contribution.toFixed(2)}</span>
      </div>
    `;
  }).join("");
}

function renderNetwork() {
  $("#networkList").innerHTML = state.selected.linked_entities.map((entity) => `
    <div class="entity ${entity.severity}"><span>${esc(entity.label)}</span><strong>${entity.value}</strong></div>
  `).join("");

  const reusedTypes = state.selected.linked_entities.filter((item) => item.value >= 2).length;
  $("#networkSummary").textContent = reusedTypes >= 2
    ? "Multiple identifiers recur across refunding accounts. Verify ownership and item evidence before release."
    : "Entity reuse is limited. The decision is driven more by behavior and refund economics than a coordinated cluster.";

  $("#timeline").innerHTML = state.selected.timeline.map((event) => `
    <div><span>${esc(event.step)}</span><strong>${esc(event.value_type === "probability" ? fmtRisk(Number(event.value) * 100) : event.value_type === "timestamp" ? fmtTimestamp(event.value) : event.value)}</strong></div>
  `).join("");
}

function renderCase() {
  renderDecision();
  renderEvidence();
  renderNetwork();
}

function renderPortfolio() {
  $("#verticalTable").innerHTML = state.data.summary.by_vertical.map((row) => `
    <div class="portfolio-row"><strong>${esc(row.vertical)}</strong><span>${fmtInr(row.refund_exposure)}</span><span class="preventable">${fmtInr(row.flagged_loss_exposure)}</span></div>
  `).join("");

  const bands = state.data.summary.risk_bands;
  const total = Object.values(bands).reduce((sum, count) => sum + count, 0);
  $("#riskDistribution").innerHTML = Object.entries(bands).map(([name, count]) => `
    <div class="risk-segment ${name}" style="width:${(count / total) * 100}%">${count}</div>
  `).join("");

  const labels = { approve: "Approve", manual_review: "Manual review", verify_evidence: "Verify evidence" };
  const legend = `<div class="distribution-legend">${Object.entries(bands).map(([name, count]) => `<div><span>${labels[name]}</span><strong>${count}</strong></div>`).join("")}</div>`;
  const actions = Object.entries(state.data.summary.by_action).map(([name, count]) => `<div class="action-row"><span>${esc(name)}</span><strong>${count}</strong></div>`).join("");
  $("#actionMix").innerHTML = legend + actions;
}

function selectedMetric() {
  return state.data.metrics.find((item) => Math.abs(item.threshold - state.threshold) < 1e-7) ?? state.data.metrics.find((item) => item.is_locked) ?? state.data.metrics[0];
}

function renderPolicyControls() {
  $("#thresholdButtons").innerHTML = state.data.metrics.map((metric) => `
    <button type="button" class="${Math.abs(metric.threshold - state.threshold) < 1e-7 ? "is-active" : ""} ${metric.is_locked ? "locked" : ""}" data-threshold="${metric.threshold}">${fmtRisk(metric.threshold_percent)}</button>
  `).join("");
  $$("#thresholdButtons button").forEach((button) => button.addEventListener("click", () => {
    state.threshold = Number(button.dataset.threshold);
    renderSummary();
    renderPolicy();
  }));
}

function renderPolicy() {
  renderPolicyControls();
  const metric = selectedMetric();
  const locked = state.data.metrics.find((item) => item.is_locked);
  $("#policySavings").textContent = fmtInr(metric.net_value);
  const delta = metric.net_value - locked.net_value;
  $("#policyDelta").textContent = metric.is_locked ? "Validation-locked operating point" : `${delta >= 0 ? "+" : ""}${fmtInr(delta)} vs locked policy`;

  const cards = [
    ["Precision", pct(metric.precision)],
    ["Recall", pct(metric.recall)],
    ["False-positive cost", fmtInr(metric.false_positive_cost)],
    [`Review cost · ${metric.review_volume} cases`, fmtInr(metric.review_cost)],
  ];
  $("#metricCards").innerHTML = cards.map(([label, value]) => `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`).join("");
  drawChart();
}

function drawChart() {
  const canvas = $("#metricsChart");
  const ratio = window.devicePixelRatio || 1;
  const bounds = canvas.getBoundingClientRect();
  const cssWidth = Math.max(bounds.width, 620);
  const cssHeight = 320;
  canvas.width = cssWidth * ratio;
  canvas.height = cssHeight * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);

  const metrics = state.data.metrics;
  const pad = { top: 34, right: 26, bottom: 45, left: 72 };
  const maxSavings = Math.max(...metrics.map((item) => item.net_value)) * 1.14;
  const minSavings = Math.min(0, ...metrics.map((item) => item.net_value));
  const chartWidth = cssWidth - pad.left - pad.right;
  const chartHeight = cssHeight - pad.top - pad.bottom;
  const scaleX = (index) => pad.left + (index / Math.max(metrics.length - 1, 1)) * chartWidth;
  const scaleY = (value) => pad.top + chartHeight - ((value - minSavings) / (maxSavings - minSavings || 1)) * chartHeight;

  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = "#fbfaf6";
  ctx.fillRect(0, 0, cssWidth, cssHeight);
  ctx.font = "11px system-ui";
  ctx.textBaseline = "middle";

  for (let tick = 0; tick <= 3; tick += 1) {
    const value = minSavings + ((maxSavings - minSavings) * tick) / 3;
    const y = scaleY(value);
    ctx.strokeStyle = "#d9ddd6";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(cssWidth - pad.right, y); ctx.stroke();
    ctx.fillStyle = "#69756e";
    ctx.fillText(`INR ${Math.round(value / 1000)}k`, 8, y);
  }

  ctx.strokeStyle = "#3157ed";
  ctx.lineWidth = 3;
  ctx.beginPath();
  metrics.forEach((item, index) => {
    const x = scaleX(index); const y = scaleY(item.net_value);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  metrics.forEach((item, index) => {
    const x = scaleX(index); const y = scaleY(item.net_value);
    const selected = item.threshold === state.threshold;
    ctx.fillStyle = selected ? "#0d2c21" : "#fbfaf6";
    ctx.strokeStyle = selected ? "#0d2c21" : "#3157ed";
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, selected ? 7 : 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#4e5c54";
    ctx.textAlign = "center";
    ctx.fillText(`${item.threshold_percent}%`, x, cssHeight - 18);
  });
  ctx.textAlign = "start";
}

function moveSelection(offset) {
  const cases = visibleCases();
  if (!cases.length) return;
  const index = Math.max(0, cases.findIndex((item) => item.case_id === state.selected.case_id));
  state.selected = cases[(index + offset + cases.length) % cases.length];
  renderQueue();
  renderCase();
}

function showPacket() {
  const item = state.selected;
  $("#packetBody").innerHTML = `
    <div class="packet-case"><div><strong>${esc(item.case_id)}</strong><span>${esc(item.merchant)} / ${esc(item.customer)}</span></div><strong>${fmtRisk(item.risk_percent)}</strong></div>
    <div class="packet-action"><span>Recommended action</span><strong>${esc(item.action)}</strong></div>
    <p>${esc(item.notes)}</p>
    <ol class="packet-evidence">${item.evidence.map((e) => `<li>${esc(e.name)}: ${e.contribution > 0 ? "+" : ""}${Number(e.contribution).toFixed(2)} log-odds contribution</li>`).join("")}</ol>
  `;
  $("#packetDialog").showModal();
}

function downloadPacket() {
  const item = state.selected;
  const packet = {
    generated_at: new Date().toISOString(),
    case_id: item.case_id,
    merchant: item.merchant,
    customer: item.customer,
    ring_probability: item.ring_probability,
    risk_percent: item.risk_percent,
    recommended_action: item.action,
    conditional_loss_if_ring: item.conditional_loss_if_ring,
    estimated_false_positive_cost: item.estimated_false_positive_cost,
    explanation_basis: item.explanation_basis,
    evidence: item.evidence,
    linked_entities: item.linked_entities,
    notes: item.notes,
  };
  const blob = new Blob([JSON.stringify(packet, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${item.case_id.toLowerCase()}-review-packet.json`;
  link.click();
  URL.revokeObjectURL(link.href);
  showToast(`Review packet downloaded for ${item.case_id}.`);
}

function switchView(viewName) {
  if (!viewName || !$(`#${viewName}View`)) return;
  $$(".nav-tab").forEach((button) => button.classList.toggle("is-active", button.dataset.view === viewName));
  $$(".view").forEach((view) => view.classList.remove("is-visible"));
  $(`#${viewName}View`).classList.add("is-visible");
  if (viewName === "policy") requestAnimationFrame(drawChart);
}

function bindEvents() {
  $$(".nav-tab[data-view]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $$("#caseFilter button").forEach((button) => button.addEventListener("click", () => {
    state.filter = button.dataset.filter;
    $$("#caseFilter button").forEach((item) => item.classList.toggle("is-active", item === button));
    const cases = visibleCases();
    if (cases.length) state.selected = cases[0];
    renderQueue(); renderCase();
  }));
  $("#caseSearch").addEventListener("input", (event) => { state.search = event.target.value; renderQueue(); });
  $("#sortBtn").addEventListener("click", () => { state.reversed = !state.reversed; renderQueue(); });
  $("#previousCase").addEventListener("click", () => moveSelection(-1));
  $("#nextCase").addEventListener("click", () => moveSelection(1));
  $("#reviewBtn").addEventListener("click", showPacket);
  $("#downloadPacket").addEventListener("click", downloadPacket);
  $("#acceptAction").addEventListener("click", () => showToast(`${state.selected.action} accepted for ${state.selected.case_id}.`));
  $("#overrideAction").addEventListener("click", () => showToast(`${state.selected.case_id} added to analyst review.`));
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== $("#caseSearch")) { event.preventDefault(); $("#caseSearch").focus(); }
  });
  window.addEventListener("resize", () => { if ($("#policyView").classList.contains("is-visible")) drawChart(); });
}

async function init() {
  const response = await fetch("/api/dashboard");
  if (!response.ok) throw new Error("Live ring-risk data unavailable");
  state.data = await response.json();
  state.threshold = state.data.active_threshold;
  state.selected = state.data.cases[0];
  renderSummary();
  renderQueue();
  renderCase();
  renderPortfolio();
  renderPolicy();
  bindEvents();
}

init().catch((error) => {
  console.error(error);
  showToast("Could not load Python-generated risk data.");
});
