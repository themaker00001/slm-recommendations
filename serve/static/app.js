const DEFAULT_ITEMS = [
  ["1", "Morton Iodized Table Salt, 26 oz | Brand: Morton"],
  ["2", "Diamond Crystal Kosher Salt, 3 lb Box | Brand: Diamond Crystal"],
  ["3", "Lay's Salt & Vinegar Potato Chips, 7.75 oz | Brand: Lay's"],
  ["4", "Himalayan Pink Salt Fine Grain, 2 lb Bag | Brand: Sherpa Pink"],
  ["5", "2% Reduced Fat Milk, Half Gallon | Brand: Horizon Organic"],
];

const OLLAMA_MODELS = ["qwen3:14b", "llama3.2:latest", "llama3:8b"];

const itemsEl = document.getElementById("items");
const resultsBody = document.getElementById("results-body");
const modelSelect = document.getElementById("teacher-model");

let itemRowId = 0;

function addItemRow(itemId, text) {
  const row = document.createElement("div");
  row.className = "item-row";
  row.dataset.itemId = itemId ?? `item-${itemRowId++}`;
  row.innerHTML = `
    <input type="text" value="${escapeHtml(text ?? "")}" placeholder="item text, e.g. brand + title">
    <button title="remove">✕</button>
  `;
  row.querySelector("button").addEventListener("click", () => row.remove());
  itemsEl.appendChild(row);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function currentItems() {
  return [...itemsEl.querySelectorAll(".item-row")]
    .map((row) => ({
      item_id: row.dataset.itemId,
      item_text: row.querySelector("input").value.trim(),
    }))
    .filter((it) => it.item_text.length > 0);
}

function currentQuery() {
  return document.getElementById("query").value.trim();
}

// --- results table, keyed by item_id so SLM and teacher runs merge ---
let resultRows = new Map();

function ensureRow(itemId, itemText) {
  if (!resultRows.has(itemId)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="item-text"></td>
      <td class="slm-label">—</td>
      <td class="slm-score">—</td>
      <td class="teacher-label">—</td>
      <td class="teacher-reason">—</td>
    `;
    resultsBody.appendChild(tr);
    resultRows.set(itemId, tr);
  }
  const tr = resultRows.get(itemId);
  tr.querySelector(".item-text").textContent = itemText;
  return tr;
}

function labelCell(label, labelName) {
  return `<span class="label-${label}">${labelName}</span>`;
}

async function runSlm() {
  const query = currentQuery();
  const items = currentItems();
  if (!items.length) return;

  const resp = await fetch("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, items }),
  });
  const data = await resp.json();
  if (data.error) {
    alert(data.error);
    return;
  }

  // Reorder table by SLM ranking
  resultsBody.innerHTML = "";
  resultRows = new Map();
  for (const r of data.results) {
    const tr = ensureRow(r.item_id, r.item_text);
    tr.querySelector(".slm-label").innerHTML = labelCell(r.predicted_label, r.label_name);
    tr.querySelector(".slm-score").textContent = r.relevance_score.toFixed(2);
    tr.classList.toggle("row-dropped", !r.kept);
  }
}

async function runTeacher() {
  const query = currentQuery();
  const items = currentItems();
  if (!items.length) return;
  const model = modelSelect.value;

  const resp = await fetch("/api/teacher", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, items, model }),
  });
  const data = await resp.json();
  if (data.error) {
    alert(data.error);
    return;
  }

  for (const r of data.results) {
    const tr = ensureRow(r.item_id, r.item_text);
    if (r.error) {
      tr.querySelector(".teacher-label").textContent = "error";
      tr.querySelector(".teacher-reason").textContent = r.error;
    } else {
      tr.querySelector(".teacher-label").innerHTML = labelCell(r.teacher_label, r.label_name);
      tr.querySelector(".teacher-reason").textContent = r.teacher_reason;
    }
  }
}

async function refreshStatus() {
  const resp = await fetch("/api/status");
  const data = await resp.json();

  const ckpt = document.getElementById("status-checkpoint");
  ckpt.textContent = data.checkpoint_loaded ? "checkpoint: loaded" : "checkpoint: not trained yet";
  ckpt.className = "status-pill " + (data.checkpoint_loaded ? "status-ok" : "status-bad");

  const ollama = document.getElementById("status-ollama");
  ollama.textContent = data.ollama_reachable ? "ollama: reachable" : "ollama: not running";
  ollama.className = "status-pill " + (data.ollama_reachable ? "status-ok" : "status-bad");
}

function init() {
  DEFAULT_ITEMS.forEach(([id, text]) => addItemRow(id, text));
  OLLAMA_MODELS.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modelSelect.appendChild(opt);
  });

  document.getElementById("add-item").addEventListener("click", () => addItemRow());
  document.getElementById("run-slm").addEventListener("click", runSlm);
  document.getElementById("run-teacher").addEventListener("click", runTeacher);

  refreshStatus();
}

init();
