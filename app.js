const state = {
  data: null,
  selected: null,
  filter: "all",
  search: "",
  sort: "risk_desc",
  threshold: null,
  activeView: "casework",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const fmtInr = (value) => `INR ${Number(value).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
const pct = (value) => `${Math.round(Number(value) * 100)}%`;
const fmtThreshold = (value) => `${Number(value).toFixed(2).replace(/0+$/, "").replace(/\.$/, "")}%`;
const fmtMetricPct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const fmtRisk = (value) => {
  const percent = Number(value);
  if (percent >= 99.95) return ">99.9%";
  if (percent < 0.05) return "<0.1%";
  if (percent >= 99) return `${percent.toFixed(3)}%`;
  if (percent < 1) return `${percent.toFixed(2)}%`;
  return percent < 10 ? `${percent.toFixed(1)}%` : `${Math.round(percent)}%`;
};
const fmtTimestamp = (value) => new Date(value).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
const cssClass = (text) => String(text).toLowerCase().replaceAll(" ", "-");
const titleCase = (text) => String(text).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
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
    const textMatch = !query || [item.case_id, item.merchant, item.customer, item.vertical, item.reason].join(" ").toLowerCase().includes(query);
    return filterMatch && textMatch;
  });
  const [field, direction] = state.sort.split("_");
  const value = field === "exposure" ? (item) => Number(item.refund_amount) : (item) => Number(item.ring_probability);
  return [...cases].sort((left, right) => {
    const difference = value(left) - value(right);
    if (difference !== 0) return direction === "asc" ? difference : -difference;
    return left.case_id.localeCompare(right.case_id);
  });
}

function renderSummary() {
  const summary = state.data.summary;
  const validation = state.data.evaluation.validation;
  const test = state.data.evaluation.test;
  const chrome = {
    casework: {
      kicker: "Decision operations / coordinated refund-abuse rings",
      title: "Refund risk operations",
      deck: "<em>Signals are cheap.</em> Defensible intervention is not.",
      aside: "High-confidence refund triage",
      cards: [
        ["Scored queue", state.data.cases.length.toLocaleString("en-IN"), "Highest-ranked final-test requests"],
        ["Queue exposure", fmtInr(summary.queue_refund_exposure), "Requested refunds in the visible queue"],
        ["Review candidates", summary.flagged_requests, "Locked-policy flags in final test"],
        ["Review threshold", fmtThreshold(state.data.active_threshold_percent), "Locked on validation only"],
      ],
    },
    portfolio: {
      kicker: "Portfolio intelligence / final synthetic test",
      title: "Refund exposure and coverage",
      deck: "<em>Confidence without coverage is incomplete.</em> Both are reported here.",
      aside: "Untouched chronological benchmark",
      cards: [
        ["Held-out requests", summary.held_out_requests.toLocaleString("en-IN"), "Final synthetic test window"],
        ["Refund exposure", fmtInr(summary.held_out_refund_exposure), "All final-test refund requests"],
        ["Conditional loss on flags", fmtInr(summary.flagged_loss_exposure), "Synthetic, conditional estimate"],
        ["Review rate", fmtMetricPct(test.flag_rate), `${test.review_volume} of ${test.rows.toLocaleString("en-IN")} requests`],
      ],
    },
    policy: {
      kicker: "Policy design / later validation window",
      title: "Precision, recall, and review capacity",
      deck: "<em>A threshold is a business decision.</em> The model only supplies probabilities.",
      aside: "Validation-locked operating policy",
      cards: [
        ["Validation requests", validation.rows.toLocaleString("en-IN"), "Later validation policy window"],
        ["Abuse requests", validation.positive_requests, "Synthetic positives in this window"],
        ["Locked-policy flags", validation.review_volume, `${validation.false_positives} false alerts`],
        ["Precision floor", ">=85%", "Declared before final-test evaluation"],
      ],
    },
  }[state.activeView];
  $("#pageKicker").textContent = chrome.kicker;
  $("#page-title").textContent = chrome.title;
  $("#pageDeck").innerHTML = chrome.deck;
  $("#viewAsideTitle").textContent = chrome.aside;
  $("#viewAsideMeta").innerHTML = `As of <strong>${fmtTimestamp(state.data.as_of)}</strong>`;
  chrome.cards.forEach(([label, value, note], index) => {
    $(`#summaryLabel${index}`).textContent = label;
    $(`#summaryValue${index}`).textContent = value;
    $(`#summaryNote${index}`).textContent = note;
  });
  $("#reviewBtn").hidden = state.activeView !== "casework";
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
      <span class="case-identity"><strong>${esc(item.case_id)}</strong><span>${esc(item.merchant)} / ${esc(titleCase(item.vertical))}</span></span>
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
  const accepted = item.analyst_action === "accepted_recommendation";
  const escalated = item.analyst_action === "escalated";
  const statusLabel = accepted ? "Recommendation accepted" : escalated ? "Sent to analyst" : item.action === "Verify evidence" ? "Evidence required" : item.action === "Manual review" ? "Review threshold met" : "Below review threshold";
  $("#caseMerchant").textContent = `${item.merchant} / ${item.vertical}`;
  $("#caseTitle").textContent = `${item.case_id} / ${item.customer}`;
  $("#scoreText").textContent = fmtRisk(item.risk_percent);
  $("#scoreMarker").style.left = `calc(${item.risk_percent}% - 1px)`;
  $("#confidenceText").textContent = "Calibrated coordinated-ring probability";
  $("#caseStatus").textContent = statusLabel;
  $("#caseStatus").className = `risk-label ${accepted ? "accepted" : escalated ? "escalated" : cssClass(item.action)}`;
  $("#caseReason").textContent = item.reason;
  $("#actionText").textContent = item.action;
  $("#caseNotes").textContent = item.notes;
  $("#refundAmount").textContent = fmtInr(item.refund_amount);
  $("#expectedLoss").textContent = fmtInr(item.conditional_loss_if_ring);
  $("#fpCost").textContent = fmtInr(item.estimated_false_positive_cost);
  $("#refundMethod").textContent = item.refund_method;
  $("#caseAge").textContent = `Scored ${fmtTimestamp(item.event_timestamp)}`;
  $("#acceptAction").textContent = accepted ? "Accepted" : "Accept recommendation";
  $("#overrideAction").textContent = escalated ? "Sent to analyst" : "Send to analyst";
  $("#acceptAction").disabled = accepted || escalated;
  $("#overrideAction").disabled = accepted || escalated;
  const outcome = $("#actionOutcome");
  outcome.hidden = !item.analyst_action;
  outcome.textContent = accepted
    ? `Recommendation accepted${item.analyst_action_at ? ` · ${fmtTimestamp(item.analyst_action_at)}` : ""}`
    : escalated ? `Sent to analyst${item.analyst_action_at ? ` · ${fmtTimestamp(item.analyst_action_at)}` : ""}` : "";
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
  $("#actionMix").innerHTML = `<div class="distribution-legend">${Object.entries(bands).map(([name, count]) => `<div><span><i class="legend-swatch ${name}"></i>${labels[name]}</span><strong>${count}</strong></div>`).join("")}</div>`;

  const test = state.data.evaluation.test;
  const metrics = [
    ["Precision", fmtMetricPct(test.precision)],
    ["Request recall", fmtMetricPct(test.recall)],
    ["Early ring recall", fmtMetricPct(test.early_ring_recall)],
    ["Review rate", fmtMetricPct(test.flag_rate)],
  ];
  $("#portfolioEvaluation").innerHTML = metrics.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  $("#coverageNarrative").textContent = `${test.true_positives} of ${test.positive_requests} abuse requests were flagged; ${test.false_negatives} were missed. ${test.early_detected_rings} of ${test.actual_rings} simulated rings were detected before half their loss. These are synthetic benchmark results, not production performance.`;
}

function selectedMetric() {
  return state.data.metrics.find((item) => Math.abs(item.threshold - state.threshold) < 1e-7) ?? state.data.metrics.find((item) => item.is_locked) ?? state.data.metrics[0];
}

function renderPolicyControls() {
  $("#thresholdButtons").innerHTML = state.data.metrics.map((metric) => `
    <button type="button" class="${Math.abs(metric.threshold - state.threshold) < 1e-7 ? "is-active" : ""} ${metric.is_locked ? "locked" : ""}" data-threshold="${metric.threshold}">${fmtThreshold(metric.threshold_percent)}</button>
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
  const metadata = state.data.policy_metadata;
  $("#thresholdLockNote").textContent = `Locked at ${fmtThreshold(locked.threshold_percent)} on ${metadata.source}: maximize recall subject to at least ${pct(metadata.precision_floor)} precision and ${metadata.minimum_flags}+ flags.`;
  $("#policySavings").textContent = fmtInr(metric.net_value);
  const delta = metric.net_value - locked.net_value;
  $("#policyDelta").textContent = metric.is_locked
    ? "Locked: meets the declared precision floor"
    : `${metric.meets_precision_floor ? "Meets" : "Fails"} 85% precision floor · ${delta >= 0 ? "+" : ""}${fmtInr(delta)} vs locked`;
  $("#policyDelta").className = metric.meets_precision_floor ? "policy-pass" : "policy-fail";
  $("#policyInterpretation").textContent = `${metric.true_positives} of ${metric.positive_requests} abuse requests caught; ${metric.false_negatives} missed; ${metric.false_positives} false alerts. Synthetic validation scenario.`;

  const cards = [
    ["Precision", fmtMetricPct(metric.precision)],
    ["Request recall", fmtMetricPct(metric.recall)],
    ["Caught", `${metric.true_positives} / ${metric.positive_requests}`],
    ["Missed abuse requests", metric.false_negatives],
    ["False-positive cost", fmtInr(metric.false_positive_cost)],
    [`Review load · ${metric.review_volume} flags`, fmtInr(metric.review_cost)],
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

async function recordAnalystAction(action, successMessage) {
  const response = await fetch(`/api/decisions/${encodeURIComponent(state.selected.case_id)}/action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, note: "Recorded from the MarginShield casework UI" }),
  });
  if (!response.ok) {
    showToast("Could not record analyst action.");
    return;
  }
  const result = await response.json();
  state.selected.analyst_action = result.action;
  state.selected.analyst_action_at = result.created_at;
  renderDecision();
  renderQueue();
  showToast(successMessage);
}

function switchView(viewName, updateHash = true) {
  if (!viewName || !$(`#${viewName}View`)) return;
  state.activeView = viewName;
  $$(".nav-tab").forEach((button) => button.classList.toggle("is-active", button.dataset.view === viewName));
  $$(".view").forEach((view) => view.classList.remove("is-visible"));
  $(`#${viewName}View`).classList.add("is-visible");
  renderSummary();
  if (updateHash) window.history.replaceState(null, "", `#${viewName}`);
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
  $("#caseSort").addEventListener("change", (event) => { state.sort = event.target.value; renderQueue(); });
  $("#previousCase").addEventListener("click", () => moveSelection(-1));
  $("#nextCase").addEventListener("click", () => moveSelection(1));
  $("#reviewBtn").addEventListener("click", showPacket);
  $("#downloadPacket").addEventListener("click", downloadPacket);
  $("#acceptAction").addEventListener("click", () => recordAnalystAction("accepted_recommendation", `${state.selected.action} accepted for ${state.selected.case_id}.`));
  $("#overrideAction").addEventListener("click", () => recordAnalystAction("escalated", `${state.selected.case_id} added to analyst review.`));
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
  renderQueue();
  renderCase();
  renderPortfolio();
  renderPolicy();
  bindEvents();
  const requestedView = window.location.hash.slice(1);
  switchView(["casework", "portfolio", "policy"].includes(requestedView) ? requestedView : "casework", false);
}

window.marginShieldChatContext = () => ({
  view: state.activeView,
  case_id: state.activeView === "casework" ? state.selected?.case_id ?? null : null,
  ring_id: null,
});

init().catch((error) => {
  console.error(error);
  showToast("Could not load Python-generated risk data.");
});
