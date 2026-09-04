const state = { rings: [], selected: null, detail: null, meta: null };
const $ = (selector) => document.querySelector(selector);
const fmtPct = (value) => {
  const percent = Number(value) * 100;
  if (percent >= 99.95) return ">99.9%";
  if (percent < 0.05) return "<0.1%";
  if (percent >= 99) return `${percent.toFixed(1)}%`;
  if (percent < 1) return `${percent.toFixed(2)}%`;
  return percent < 10 ? `${percent.toFixed(1)}%` : `${Math.round(percent)}%`;
};
const esc = (value) => String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));

function renderSummary() {
  const rings = state.rings;
  $("#ringCount").textContent = rings.length;
  $("#linkedReviewCases").textContent = rings.reduce((sum, ring) => sum + ring.high_risk_case_count, 0);
  $("#largestLink").textContent = Math.max(...rings.map((ring) => ring.max_linked_accounts), 0);
  $("#windowDays").textContent = `${state.meta.window_days}d`;
  $("#ringListCount").textContent = rings.length;
  $("#asOfText").textContent = `As of ${new Date(state.meta.as_of).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" })}`;
  $("#rankingMethod").textContent = state.meta.ranking_note;
}

function renderList() {
  $("#ringList").innerHTML = state.rings.map((ring) => `
    <button class="ring-row ${state.selected === ring.ring_id ? "is-active" : ""}" data-ring="${ring.ring_id}" type="button">
      <span class="ring-row-top"><strong>${ring.ring_id}</strong><b>P${ring.queue_rank}</b></span>
      <span>${ring.case_count} requests · ${ring.entity_count} shared identifiers</span>
      <small>${ring.high_risk_case_count} meet policy · ${Number(ring.model_weighted_exposure_inr).toLocaleString("en-IN")} INR model-weighted exposure</small>
    </button>
  `).join("");
  document.querySelectorAll(".ring-row").forEach((button) => button.addEventListener("click", () => selectRing(button.dataset.ring)));
}

function graphPositions(nodes) {
  const cases = nodes.filter((node) => node.kind === "case");
  const entities = nodes.filter((node) => node.kind === "entity");
  const positions = new Map();
  entities.forEach((node, index) => positions.set(node.id, { x: 500, y: 135 + index * (290 / Math.max(entities.length - 1, 1)) }));
  cases.forEach((node, index) => {
    const side = index % 2 === 0 ? 225 : 775;
    const row = Math.floor(index / 2);
    positions.set(node.id, { x: side, y: 80 + row * (400 / Math.max(Math.ceil(cases.length / 2) - 1, 1)) });
  });
  return positions;
}

function renderGraph(detail) {
  const positions = graphPositions(detail.nodes);
  const svg = $("#ringGraph");
  const edgeMarkup = detail.edges.map((edge) => {
    const source = positions.get(edge.source); const target = positions.get(edge.target);
    return `<line x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" class="graph-edge ${edge.relation.replace(" ", "-")}" />`;
  }).join("");
  const nodeMarkup = detail.nodes.map((node) => {
    const point = positions.get(node.id);
    if (node.kind === "entity") return `<g class="graph-node entity-node" transform="translate(${point.x} ${point.y})"><circle r="39"></circle><text y="-2">${esc(node.entity_type)}</text><text y="15">${node.accounts} accounts</text></g>`;
    const riskClass = node.risk_probability >= state.meta.verify_evidence_threshold ? "verify" : node.risk_probability >= state.meta.manual_review_threshold ? "review" : "approve";
    return `<g class="graph-node case-node ${riskClass}" transform="translate(${point.x} ${point.y})"><circle r="31"></circle><text y="-3">${esc(node.label)}</text><text y="15">${fmtPct(node.risk_probability)}</text></g>`;
  }).join("");
  svg.innerHTML = `<title>${esc(detail.ring_id)} relationship graph</title>${edgeMarkup}${nodeMarkup}`;
}

function renderDetail(detail) {
  $("#ringLabel").textContent = "Suspected abuse ring";
  $("#ringHeading").textContent = detail.ring_id;
  $("#ringRank").textContent = `Queue rank P${detail.queue_rank}`;
  $("#ringFacts").innerHTML = [
    ["Connected requests", detail.case_count],
    ["Shared identifiers", detail.entity_count],
    ["Mean model risk", fmtPct(detail.mean_risk_probability)],
    ["Model-weighted exposure", `INR ${Number(detail.model_weighted_exposure_inr).toLocaleString("en-IN")}`],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  $("#ringReasons").innerHTML = detail.reasons.map((reason) => `<li>${esc(reason)}</li>`).join("");
  const cases = detail.nodes.filter((node) => node.kind === "case");
  $("#ringCases").innerHTML = cases.map((item) => `<div class="ring-case-row"><span><strong>${esc(item.label)}</strong><small>${esc(item.merchant_id)}</small></span><span>INR ${Number(item.refund_amount_inr).toLocaleString("en-IN")}</span><b>${fmtPct(item.risk_probability)}</b></div>`).join("");
  renderGraph(detail);
}

async function selectRing(ringId) {
  state.selected = ringId;
  renderList();
  const response = await fetch(`/api/rings/${encodeURIComponent(ringId)}`);
  if (!response.ok) throw new Error("Ring detail unavailable");
  state.detail = await response.json();
  renderDetail(state.detail);
}

async function init() {
  const response = await fetch("/api/rings");
  if (!response.ok) throw new Error("Ring monitor unavailable");
  state.meta = await response.json();
  state.rings = state.meta.rings;
  renderSummary(); renderList();
  if (state.rings.length) await selectRing(state.rings[0].ring_id);
}

init().catch((error) => {
  console.error(error);
  $("#ringHeading").textContent = "Graph monitor unavailable";
});
